import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import time
from collections import defaultdict, deque, Counter

# --- Imports from your project files ---
from .creat_vrp_revised_new import creat_data
from .vrpUpdate_1 import update_state, update_mask
from .VRP_SAC_Agent import AgentSAC, Encoder
from torch_geometric.data import Batch, Data

# Ensure the device is set correctly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# +++ CORRECTED VRP Environment Class (Faithful replica of the training environment) +++
class VRPEnv:
    def __init__(self, initial_batch_data, n_nodes):
        self.n_nodes = n_nodes
        self.depot_node_idx_local = n_nodes - 1
        self.batch_size = initial_batch_data.num_graphs

    def reset(self, batch_data):
        self.subtour_min_date = torch.full((self.batch_size,), -1.0, device=device)
        self.subtour_max_date = torch.full((self.batch_size,), -1.0, device=device)
        # Add full state tracking
        self.subtour_nodes_indices = [[] for _ in range(self.batch_size)]
        self.all_completed_subtour_labours = [[] for _ in range(self.batch_size)]

        self.static_capacity = batch_data.capacity.clone().to(device)
        self.scalar_static_capacity = self.static_capacity[0, 0].item()
        self.demands = batch_data.demand.clone().view(self.batch_size, self.n_nodes).to(device)
        self.capacities = self.static_capacity.clone()
        self.all_due_dates = batch_data.x[:, 4].clone().view(self.batch_size, self.n_nodes).to(device)
        self.dynamic_due_dates = self.all_due_dates.clone()
        self.all_intervals = batch_data.x[:, 1].clone().view(self.batch_size, self.n_nodes).to(device)
        self.visited_mask = torch.zeros((self.batch_size, self.n_nodes), dtype=torch.bool, device=device)

        # Correctly create the initial state with a mask
        initial_mask, self.visited_mask = update_mask(
            self.demands, self.capacities,
            torch.full((self.batch_size, 1), self.depot_node_idx_local, device=device),
            self.visited_mask, self.depot_node_idx_local, self.all_due_dates,
            self.subtour_min_date, self.subtour_max_date,
            num_completed_slots=torch.zeros(self.batch_size, device=device, dtype=torch.long),
            current_subtour_size=torch.zeros(self.batch_size, device=device, dtype=torch.long)
        )
        initial_state_list = []
        for i in range(self.batch_size):
            instance_data = batch_data[i]
            new_data_point = Data(
                x=instance_data.x, edge_index=instance_data.edge_index,
                edge_attr=instance_data.edge_attr, raw_panel=instance_data.raw_panel,
                demand=instance_data.demand, capacity=instance_data.capacity,
                mask=initial_mask[i].unsqueeze(-1)
            )
            initial_state_list.append(new_data_point)
        self.current_batch_data = Batch.from_data_list(initial_state_list).to(device)
        return self.current_batch_data

    def step(self, actions_tensor):
        # Update subtour tracking
        for b_idx in range(self.batch_size):
            selected_node_idx = actions_tensor[b_idx].item()
            if selected_node_idx == self.depot_node_idx_local:
                if len(self.subtour_nodes_indices[b_idx]) > 1:
                    subtour_labour = sum(
                        self.current_batch_data.x[self.current_batch_data.ptr[b_idx] + nid, 2].item() for nid in
                        self.subtour_nodes_indices[b_idx])
                    self.all_completed_subtour_labours[b_idx].append(subtour_labour)
                self.subtour_min_date[b_idx] = -1.0
                self.subtour_max_date[b_idx] = -1.0
                self.subtour_nodes_indices[b_idx] = []
            else:
                task_duedate = self.all_due_dates[b_idx, selected_node_idx].item()
                if self.subtour_min_date[b_idx].item() == -1.0:
                    self.subtour_min_date[b_idx] = task_duedate
                    self.subtour_max_date[b_idx] = task_duedate
                else:
                    self.subtour_min_date[b_idx] = min(self.subtour_min_date[b_idx].item(), task_duedate)
                    self.subtour_max_date[b_idx] = max(self.subtour_max_date[b_idx].item(), task_duedate)
                self.subtour_nodes_indices[b_idx].append(selected_node_idx)

        # Update core state variables
        self.capacities, self.demands, self.dynamic_due_dates = update_state(
            self.demands, self.capacities, self.dynamic_due_dates, self.all_intervals,
            self.subtour_min_date, actions_tensor.unsqueeze(-1), self.scalar_static_capacity, self.depot_node_idx_local
        )

        # Provide correct, tracked inputs to the mask function
        num_completed_slots = torch.tensor([len(s) for s in self.all_completed_subtour_labours], device=device,
                                           dtype=torch.long)
        current_subtour_size = torch.tensor([len(s) for s in self.subtour_nodes_indices], device=device,
                                            dtype=torch.long)
        next_mask, self.visited_mask = update_mask(
            self.demands, self.capacities, actions_tensor.unsqueeze(-1),
            self.visited_mask, self.depot_node_idx_local, self.all_due_dates,
            self.subtour_min_date, self.subtour_max_date,
            num_completed_slots=num_completed_slots,
            current_subtour_size=current_subtour_size
        )

        # Rebuild the state batch correctly
        next_state_list = []
        for i in range(self.batch_size):
            instance_data = self.current_batch_data[i]
            new_data_point = Data(
                x=instance_data.x, edge_index=instance_data.edge_index,
                edge_attr=instance_data.edge_attr, raw_panel=instance_data.raw_panel,
                demand=self.demands[i].unsqueeze(-1), capacity=self.capacities[i].unsqueeze(-1),
                mask=next_mask[i].unsqueeze(-1)
            )
            next_state_list.append(new_data_point)
        self.current_batch_data = Batch.from_data_list(next_state_list).to(device)
        dones = self.demands[:, :self.depot_node_idx_local].le(0).all(dim=1)
        return self.current_batch_data, torch.zeros(self.batch_size), dones


# --- Core Logic Functions (Unchanged) ---

def load_sac_model(model_path_prefix, config, n_nodes):
    print(f"Loading SAC model components from prefix: {model_path_prefix}")
    agent = AgentSAC(
        raw_node_feature_dim=5, demand_feature_dim=1,
        hidden_node_dim=config["hidden_node_dim"], input_edge_dim=config["input_edge_dim"],
        hidden_edge_dim=config["hidden_edge_dim"], conv_layers=config["conv_layers"],
        n_nodes=n_nodes, learning_rate=config["learning_rate"]
    )
    try:
        agent.encoder.load_state_dict(torch.load(f'{model_path_prefix}_encoder.pth', map_location=device))
        agent.actor.load_state_dict(torch.load(f'{model_path_prefix}_actor.pth', map_location=device))
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please ensure the model prefix is correct and all component files exist.")
        raise
    agent.encoder.eval();
    agent.actor.eval()
    print("SAC model components loaded successfully and set to evaluation mode.")
    return agent


def parse_and_filter_actions_to_slots(actions_sequence, graph_data, aircraft_id):
    slots, current_slot_tasks = [], []
    depot_node_idx = graph_data.num_nodes - 1
    skipped_slots_count = 0
    if hasattr(actions_sequence, 'cpu'): actions_sequence = actions_sequence.cpu().tolist()
    for node_idx in actions_sequence:
        node_idx = int(node_idx)
        if node_idx == depot_node_idx:
            if current_slot_tasks:
                if len(current_slot_tasks) > 1:
                    preferred_date = min(task['duedate'] for task in current_slot_tasks)
                    slots.append({"aircraft_id": aircraft_id, "preferred_date": int(preferred_date),
                                  "task_details": list(current_slot_tasks)})
                else:
                    skipped_slots_count += 1
                current_slot_tasks.clear()
        else:
            if 0 <= node_idx < graph_data.x.size(0):
                task_features = graph_data.x[node_idx]
                current_slot_tasks.append(
                    {"task_code": f"T{int(task_features[0].item())}", "interval": task_features[1].item(),
                     "labour": task_features[2].item(), "duedate": task_features[4].item(),
                     "panel": graph_data.raw_panel[node_idx]})
    return slots, skipped_slots_count


def combine_slots_within_window(aircraft_slots, window_days=60):
    if not aircraft_slots: return []
    sorted_slots = sorted(aircraft_slots, key=lambda s: s['preferred_date'])
    combined_slots = []
    current_combined_slot = sorted_slots[0]
    for next_slot in sorted_slots[1:]:
        if next_slot['preferred_date'] - current_combined_slot['preferred_date'] <= window_days:
            current_combined_slot['task_details'].extend(next_slot['task_details'])
        else:
            combined_slots.append(current_combined_slot)
            current_combined_slot = next_slot
    combined_slots.append(current_combined_slot)
    return combined_slots


def calculate_final_kpis(df_schedule):
    if df_schedule.empty: return {"avg_spillage_percentage_per_task": 0, "avg_panel_reuse_percentage": 0,
                                  "avg_labour_per_slot": 0, "variance_labour_per_slot": 0, "num_slots_with_gt1_task": 0,
                                  "num_unique_task_visits": 0}
    total_spillage_percentage, total_tasks_in_schedule, panel_reuse_percentages = 0, 0, []
    for _, row in df_schedule.iterrows():
        assigned_date, task_details = row['assigned_date'], row['task_details']
        total_tasks_in_schedule += len(task_details)
        for task in task_details:
            if task['interval'] > 0: total_spillage_percentage += (abs(task['duedate'] - assigned_date) / task[
                'interval']) * 100
        panel_counts = Counter(task['panel'] for task in task_details if
                               isinstance(task['panel'], str) and task['panel'] not in ["0", "NOTE", "DEPOT"])
        all_panels_in_slot = set(panel_counts.keys())
        if all_panels_in_slot:
            num_reused = sum(1 for count in panel_counts.values() if count > 1)
            panel_reuse_percentages.append((num_reused / len(all_panels_in_slot)) * 100)
    avg_spillage = total_spillage_percentage / total_tasks_in_schedule if total_tasks_in_schedule > 0 else 0
    unique_tasks = df_schedule.explode('task_details')['task_details'].apply(lambda x: x['task_code']).nunique()
    return {"avg_spillage_percentage_per_task": avg_spillage,
            "avg_panel_reuse_percentage": np.mean(panel_reuse_percentages) if panel_reuse_percentages else 0,
            "avg_labour_per_slot": df_schedule['total_labour'].mean(),
            "variance_labour_per_slot": df_schedule['total_labour'].var(), "num_slots_with_gt1_task": len(df_schedule),
            "num_unique_task_visits": unique_tasks}


def generate_and_schedule():
    # --- 1. CONFIGURATION (Must match the trained SAC model) ---
    config = {
        "learning_rate": 1e-5, "hidden_node_dim": 256, "hidden_edge_dim": 32,
        "input_edge_dim": 1, "conv_layers": 3, "target_task_nodes": 361,
        "n_aircraft": 1, "k_nearest_neighbors": 10,
        "csv_file_path": "final_imputed_maintenance_policy.csv", "max_episode_len": 1500,
    }

    # !!! IMPORTANT: UPDATE THIS LINE !!!
    MODEL_PATH_PREFIX = "saved_models_sac/best_model_step_70000"

    # --- 2. GENERATE & PRE-PROCESS PLANS ---
    n_nodes = config["target_task_nodes"] + 1
    agent = load_sac_model(MODEL_PATH_PREFIX, config, n_nodes)
    num_aircraft = 31
    all_schedulable_slots, total_skipped = [], 0

    print(f"\nGenerating and pre-processing plans for {num_aircraft} aircraft...")
    for i in range(num_aircraft):
        aircraft_id_to_process = i + 1
        print(f"  Processing Aircraft ID: {i + 1}/{num_aircraft}")
        instance_loader, _ = creat_data(file_path=config["csv_file_path"], num_samples=1, batch_size=1, n_aircraft=1,
                                        target_task_nodes=config["target_task_nodes"],
                                        k_nearest_neighbors=config["k_nearest_neighbors"],aircraft_id=aircraft_id_to_process)
        graph_data_batch = next(iter(instance_loader))

        env = VRPEnv(graph_data_batch, n_nodes)
        state = env.reset(graph_data_batch)
        actions_sequence = []
        for _ in range(config["max_episode_len"]):
            actions_np, _ = agent.select_action(state)
            actions_sequence.append(actions_np[0])
            state, _, dones = env.step(torch.tensor(actions_np, device=device))
            if dones.all(): break

        initial_aircraft_slots, skipped_count = parse_and_filter_actions_to_slots(actions_sequence, graph_data_batch[0],
                                                                                  aircraft_id=i + 1)
        total_skipped += skipped_count
        combined_aircraft_slots = combine_slots_within_window(initial_aircraft_slots)
        all_schedulable_slots.extend(combined_aircraft_slots)

    # --- 3. ADVANCED CONFLICT RESOLUTION ---
    print("\nStarting advanced conflict resolution on consolidated slots...")
    slots_by_day = defaultdict(list)
    for slot in all_schedulable_slots: slots_by_day[slot['preferred_date']].append(slot)
    calendar, final_schedule_list = {}, []
    for date in sorted(slots_by_day.keys()):
        competing_slots = slots_by_day[date]
        competing_slots.sort(key=lambda s: len(s['task_details']), reverse=True)
        for slot in competing_slots:
            date_to_check = slot['preferred_date']
            while date_to_check in calendar: date_to_check -= 1
            calendar[date_to_check] = slot
            final_slot_details = slot.copy()
            final_slot_details.update(
                {'assigned_date': date_to_check, 'days_moved': slot['preferred_date'] - date_to_check,
                 'tasks': [t['task_code'] for t in slot['task_details']],
                 'total_labour': sum(t['labour'] for t in slot['task_details'])})
            final_schedule_list.append(final_slot_details)
    df_schedule = pd.DataFrame(final_schedule_list)
    print("Conflict resolution complete.")

    # --- 4. CALCULATE AND DISPLAY FINAL KPIs ---
    final_kpis = calculate_final_kpis(df_schedule)
    print("\n" + "=" * 65)
    print("FLEET-WIDE KPIs (Calculated on Final Conflict-Resolved Schedule)")
    print("=" * 65)
    print(f"  Total Consolidated Maintenance Slots: {final_kpis['num_slots_with_gt1_task']:.0f}")
    print(f"  Total Unique Tasks Scheduled        : {final_kpis['num_unique_task_visits']:.0f}")
    print(f"  Avg. Spillage per Task (vs Interval): {final_kpis['avg_spillage_percentage_per_task']:.2f}%")
    print(f"  Avg. Panel Reuse Percentage         : {final_kpis['avg_panel_reuse_percentage']:.2f}%")
    print(f"  Avg. Labour per Consolidated Slot   : {final_kpis['avg_labour_per_slot']:.2f} hrs")
    print(f"  Variance of Labour                  : {final_kpis['variance_labour_per_slot']:.2f}")
    print(f"  Total Out-of-Phase Slots Skipped    : {total_skipped}")
    print("=" * 65 + "\n")

    # --- 5. DISPLAY PER-AIRCRAFT SCHEDULES ---
    if not df_schedule.empty:
        df_schedule['tasks_str'] = df_schedule['tasks'].apply(lambda x: ', '.join(x))
        print("\n" + "=" * 60 + "\nFINAL MAINTENANCE SCHEDULES BY AIRCRAFT\n" + "=" * 60)
        for aircraft_id, group in df_schedule.sort_values(by='assigned_date').groupby('aircraft_id'):
            print(f"\n--- AIRCRAFT ID: {aircraft_id} ---")
            display_group = group[['assigned_date', 'days_moved', 'total_labour', 'tasks_str']]
            print(display_group.to_string(index=False))
    else:
        print("No schedulable (multi-task) slots were generated for any aircraft.")


if __name__ == '__main__':
    if not os.path.exists("Maintenance policy data.csv"):
        print("Warning: 'Maintenance policy data.csv' not found. Creating a dummy CSV.")
        pd.DataFrame({'Task_code': [f'T{i:03}' for i in range(1, 300)], 'Interval': np.random.randint(30, 365, 299),
                      'Labour': np.random.randint(5, 100, 299), 'Skill': [f'S{(i % 4) + 1}' for i in range(299)],
                      'Panel': [f'P{(i % 20) + 1}' for i in range(299)]}).to_csv("Maintenance policy data.csv",
                                                                                 index=False)

    generate_and_schedule()