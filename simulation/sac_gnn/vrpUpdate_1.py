# vrpUpdate.py

import torch
import time
import torch.nn.functional as F


def update_state(demand, dynamic_capacity, dynamic_due_dates, all_intervals, current_slot_date, selected, c,
                 depot_idx_local):
    """
    Updates demand, capacity, and crucially, the next due date for recurring tasks.
    """
    selected_squeezed = selected.squeeze(-1)
    is_depot_selected_batch = selected_squeezed.eq(depot_idx_local)

    new_demand = demand.clone()
    new_dynamic_capacity = dynamic_capacity.clone()
    new_dynamic_due_dates = dynamic_due_dates.clone()

    for b_idx in range(demand.size(0)):
        if is_depot_selected_batch[b_idx]:
            # Reset capacity when visiting the depot
            new_dynamic_capacity[b_idx] = c
        else:
            # Update state for a task visit
            node_idx = selected_squeezed[b_idx].item()

            if new_demand[b_idx, node_idx] > 0:
                new_demand[b_idx, node_idx] -= 1

            # Calculate and set the next due date for this task
            task_interval = all_intervals[b_idx, node_idx]
            if current_slot_date[b_idx] != -1:
                new_dynamic_due_dates[b_idx, node_idx] = current_slot_date[b_idx] + task_interval

    return new_dynamic_capacity, new_demand, new_dynamic_due_dates



def update_mask(demands, capacities, current_nodes, visited_mask, depot_node_idx, 
                all_due_dates, 
                subtour_min_date, subtour_max_date, # Use dynamic min/max dates
                num_completed_slots, current_subtour_size,
                max_multi_task_slots=12,
                # NEW: Max allowed spillage for an entire slot (e.g., 7 days)
                max_allowed_slot_spillage=40):
    """
    Updates the mask for visitable nodes.
    - REVISED: Masks tasks if adding them would exceed the max_allowed_slot_spillage.
    """
    batch_size, n_nodes = demands.shape
    device = demands.device

    mask = torch.zeros((batch_size, n_nodes), dtype=torch.bool, device=device)

    # Standard Masking
    mask.copy_(visited_mask)
    mask = mask | (demands > capacities)
    demand_satisfied = demands <= 0
    demand_satisfied[:, depot_node_idx] = False
    mask = mask | demand_satisfied

    # --- NEW DYNAMIC SPILLAGE MASK ---
    is_building_subtour = (current_subtour_size > 0).view(-1, 1)

    # Calculate what the new min/max dates would be if we added each task
    # For tasks already in the slot, the range won't change. For new tasks, it might expand.
    potential_min_dates = torch.minimum(subtour_min_date.view(-1, 1), all_due_dates)
    potential_max_dates = torch.maximum(subtour_max_date.view(-1, 1), all_due_dates)
    
    # Calculate the potential total date range of the slot for each candidate task
    potential_slot_ranges = potential_max_dates - potential_min_dates
    
    # Identify which tasks would violate the max allowed spillage for the slot
    exceeds_spillage_limit = potential_slot_ranges > max_allowed_slot_spillage
    
    # Apply this mask ONLY to instances that are currently building a subtour
    mask = mask | (exceeds_spillage_limit & is_building_subtour)
    # ------------------------------------

    # Hard Slot Limit Logic
    limit_reached = num_completed_slots >= max_multi_task_slots
    limit_reached_indices = torch.where(limit_reached)[0]
    if len(limit_reached_indices) > 0:
        subtour_just_started = current_subtour_size[limit_reached_indices] == 1
        force_depot_return_indices = limit_reached_indices[subtour_just_started]
        if len(force_depot_return_indices) > 0:
            mask[force_depot_return_indices, :depot_node_idx] = True

    # Final Safety Checks
    is_at_depot = (current_nodes == depot_node_idx).squeeze(-1)
    visited_mask[is_at_depot, :depot_node_idx] = 0
    mask.scatter_(1, current_nodes, 1)

    all_tasks_masked = mask[:, :depot_node_idx].all(dim=1)
    mask[all_tasks_masked, depot_node_idx] = False

    return mask, visited_mask