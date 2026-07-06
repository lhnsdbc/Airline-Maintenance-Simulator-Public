# MODIFIED SCRIPT to handle specific initial states from a CSV file

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from tqdm import tqdm
import pandas as pd

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:
    from torch_geometric.data import DataLoader as PyGDataLoader


def load_maintenance_data(file_path):
    df = pd.read_csv(file_path)
    df['Skill_Encoded'] = pd.factorize(df['Skill'])[0]
    df['Task_Code_Encoded'] = pd.factorize(df['Task_code'])[0]
    return df


def creat_instance_fleet(df_maintenance_data, target_task_nodes=361, k_nearest_neighbors=50, aircraft_id=None,
                         **kwargs):
    """
    Generates a single VRP graph instance for a specific aircraft if aircraft_id is provided.
    If aircraft_id is None, it falls back to the original behavior of random due date generation for one aircraft.
    This function ensures that the number of task nodes in the graph is exactly
    equal to 'target_task_nodes'.
    """
    potential_nodes = []

    unique_task_base_features = df_maintenance_data[['Task_Code_Encoded', 'Interval', 'Labour', 'Skill_Encoded']].values
    unique_task_panels = df_maintenance_data['Panel'].values
    n_unique_tasks = len(df_maintenance_data)
    intervals = unique_task_base_features[:, 1]

    # --- CORE MODIFICATION: Load specific due dates or randomize ---
    if aircraft_id is not None:
        # Load deterministic due dates from the specified aircraft column
        aircraft_column_name = f'Aircraft_{aircraft_id}'
        if aircraft_column_name not in df_maintenance_data.columns:
            raise ValueError(f"Column {aircraft_column_name} not found in the CSV file.")
        duedates = df_maintenance_data[aircraft_column_name].values
    else:
        # Fallback to original random generation if no aircraft_id is given
        print("Warning: No aircraft_id provided. Generating random due dates for a single instance.")
        duedates = np.random.randint(1, intervals + 1)

    for task_idx in range(n_unique_tasks):
        interval = intervals[task_idx]
        duedate = duedates[task_idx]

        # Your improved, more realistic demand calculation
        demand_val = 0
        if duedate < 365:
            demand_val = 1
            remaining_horizon = 365 - duedate
            recurring_demand = int(remaining_horizon / interval) if interval > 0 else 0
            demand_val += recurring_demand

        node_feat_vector = [
            unique_task_base_features[task_idx, 0],
            interval,
            unique_task_base_features[task_idx, 2],
            unique_task_base_features[task_idx, 3],
            float(duedate)
        ]

        potential_nodes.append({
            "features": node_feat_vector,
            "demand": float(demand_val),
            "panel": unique_task_panels[task_idx],
            "original_duedate": float(duedate)
        })

    # --- Step 2: Ensure final list of tasks matches the target size ---
    active_nodes = [node for node in potential_nodes if node["demand"] > 0]
    inactive_nodes = [node for node in potential_nodes if node["demand"] == 0]
    final_task_nodes = []
    if len(active_nodes) >= target_task_nodes:
        indices_to_sample = np.random.choice(len(active_nodes), target_task_nodes, replace=False)
        final_task_nodes = [active_nodes[i] for i in indices_to_sample]
    else:
        final_task_nodes.extend(active_nodes)
        num_padding_needed = target_task_nodes - len(active_nodes)
        if num_padding_needed > 0 and inactive_nodes:
            padding_indices = np.random.choice(len(inactive_nodes), min(num_padding_needed, len(inactive_nodes)),
                                               replace=False)
            padding_nodes = [inactive_nodes[i] for i in padding_indices]
            final_task_nodes.extend(padding_nodes)

    # --- Step 3: Build the final graph tensors from the fixed-size list ---
    all_nodes_features_list = [node["features"] for node in final_task_nodes]
    all_demands_list = [[node["demand"]] for node in final_task_nodes]
    raw_panel_list_for_instance = [node["panel"] for node in final_task_nodes]
    all_edge_indices_list = []
    all_edge_attributes_list = []
    num_task_nodes_in_graph = len(final_task_nodes)

    if num_task_nodes_in_graph > 0:
        due_dates_np = np.array([node["original_duedate"] for node in final_task_nodes]).reshape(1, -1)
        dist_matrix = np.abs(due_dates_np - due_dates_np.T)
        k = min(k_nearest_neighbors, num_task_nodes_in_graph - 1)
        for i in range(num_task_nodes_in_graph):
            neighbor_indices = np.argsort(dist_matrix[i, :])[1:k + 1]
            for neighbor_idx in neighbor_indices:
                all_edge_indices_list.append([i, neighbor_idx])
                all_edge_attributes_list.append([dist_matrix[i, neighbor_idx]])

    # Add the depot node
    depot_global_idx = num_task_nodes_in_graph
    depot_feat_vector = [0.0] * 5
    all_nodes_features_list.append(depot_feat_vector)
    all_demands_list.append([0.0])
    raw_panel_list_for_instance.append("DEPOT")

    # Connect the depot to all task nodes
    for task_node_idx in range(num_task_nodes_in_graph):
        all_edge_indices_list.append([depot_global_idx, task_node_idx])
        all_edge_attributes_list.append([0.0])
        all_edge_indices_list.append([task_node_idx, depot_global_idx])
        all_edge_attributes_list.append([0.0])

    # Convert to Tensors
    x_nodes = torch.tensor(all_nodes_features_list, dtype=torch.float)
    edge_index = torch.tensor(all_edge_indices_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(all_edge_attributes_list, dtype=torch.float)
    demand_features = torch.tensor(all_demands_list, dtype=torch.float)
    capacity_val = torch.tensor([[500.0]], dtype=torch.float)
    actual_total_nodes_in_instance = x_nodes.size(0)

    return x_nodes, edge_index, edge_attr, demand_features, capacity_val, raw_panel_list_for_instance, actual_total_nodes_in_instance


def creat_data(file_path, num_samples=1000, batch_size=32, n_aircraft=1, aircraft_id=None,
               aircraft_ids=None, **kwargs):
    df = load_maintenance_data(file_path)
    graph_data_list = []
    num_nodes_per_graph_sample = None
    if aircraft_ids is not None:
        aircraft_ids = list(aircraft_ids)
        num_samples = len(aircraft_ids)
        batch_size = min(batch_size, num_samples)

    # This outer loop remains for generating multiple samples if needed,
    # but for your validation case, num_samples will be 1.
    for i in tqdm(range(num_samples), desc="Generating graph instances"):
        current_aircraft_id = aircraft_ids[i] if aircraft_ids is not None else aircraft_id
        x, edge_idx, edge_att, demand, capacity, raw_panel, n_nodes_sample = \
            creat_instance_fleet(df, aircraft_id=current_aircraft_id, **kwargs)

        if i == 0:
            num_nodes_per_graph_sample = n_nodes_sample

        data_obj = Data(x=x, edge_index=edge_idx, edge_attr=edge_att,
                        demand=demand, capacity=capacity)
        data_obj.raw_panel = raw_panel
        graph_data_list.append(data_obj)

    pyg_dataloader = PyGDataLoader(graph_data_list, batch_size=batch_size,
                                   shuffle=False)  # Shuffle=False for validation

    if num_nodes_per_graph_sample is None and num_samples > 0:
        raise ValueError("Could not determine num_nodes_per_graph_sample.")

    return pyg_dataloader, num_nodes_per_graph_sample
