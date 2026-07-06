import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax
import numpy as np
import math
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import LambdaLR
import time
from .vrpUpdate_1 import update_mask, update_state

INIT = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_grad_norm = 2


class GatConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels,
                 negative_slope=0.2, dropout=0):
        super(GatConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.fc_node_transform = nn.Linear(in_channels, out_channels)
        self.attn_fc = nn.Linear(2 * out_channels + edge_channels, out_channels)
        if INIT:
            nn.init.orthogonal_(self.fc_node_transform.weight, gain=math.sqrt(2))
            if self.fc_node_transform.bias is not None: nn.init.zeros_(self.fc_node_transform.bias)
            nn.init.xavier_uniform_(self.attn_fc.weight, gain=math.sqrt(2))
            if self.attn_fc.bias is not None: nn.init.zeros_(self.attn_fc.bias)

    def forward(self, x, edge_index, edge_attr, size_i=None):
        x_transformed = self.fc_node_transform(x)
        return self.propagate(edge_index, x=x_transformed, edge_attr=edge_attr, size=size_i)

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        alpha_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        alpha = self.attn_fc(alpha_input)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        # if edge_attr is not None and edge_attr.size(1) > 0 and edge_attr.shape[0] == alpha.shape[0]:
        #     due_date_dist_mask = (edge_attr[:, 0] < 30.0).unsqueeze(-1).float()
        #     alpha = alpha * due_date_dist_mask
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha

    def update(self, aggr_out):
        return aggr_out


class Encoder(nn.Module):
    def __init__(self, raw_node_feature_dim, demand_feature_dim, hidden_node_dim,
                 input_edge_dim, hidden_edge_dim, conv_layers=3):
        super(Encoder, self).__init__()
        self.hidden_node_dim = hidden_node_dim
        self.fc_node = nn.Linear(raw_node_feature_dim + demand_feature_dim, hidden_node_dim)
        self.bn_node = nn.BatchNorm1d(hidden_node_dim)
        self.use_edge_features = input_edge_dim > 0
        if self.use_edge_features:
            self.fc_edge = nn.Linear(input_edge_dim, hidden_edge_dim)
            self.bn_edge = nn.BatchNorm1d(hidden_edge_dim)
        else:
            hidden_edge_dim = 0
        self.convs = nn.ModuleList()
        for _ in range(conv_layers):
            self.convs.append(GatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim))
        if INIT:
            nn.init.orthogonal_(self.fc_node.weight, gain=math.sqrt(2))
            if self.fc_node.bias is not None: nn.init.zeros_(self.fc_node.bias)
            if self.use_edge_features:
                nn.init.orthogonal_(self.fc_edge.weight, gain=math.sqrt(2))
                if self.fc_edge.bias is not None: nn.init.zeros_(self.fc_edge.bias)

    def forward(self, data: Batch):
        node_input_features = torch.cat([data.x, data.demand], dim=-1)
        x = F.relu(self.bn_node(self.fc_node(node_input_features)))
        edge_attr_transformed = None
        if self.use_edge_features and data.edge_attr is not None and data.edge_attr.numel() > 0:
            edge_attr_transformed = F.relu(self.bn_edge(self.fc_edge(data.edge_attr)))
        elif self.use_edge_features:
            num_edges = data.edge_index.size(1)
            gat_expected_edge_dim = self.convs[0].attn_fc.in_features - 2 * self.hidden_node_dim
            if gat_expected_edge_dim > 0:
                edge_attr_transformed = torch.zeros(num_edges, gat_expected_edge_dim, device=x.device)
        for conv_layer in self.convs:
            x_new = conv_layer(x, data.edge_index, edge_attr_transformed)
            x = x + F.relu(x_new)
        return x


class ProbAttention(nn.Module):
    def __init__(self, n_heads, query_input_dim, context_node_input_dim, attention_hidden_dim):
        super(ProbAttention, self).__init__()
        self.n_heads = n_heads
        self.attention_hidden_dim = attention_hidden_dim
        assert attention_hidden_dim % n_heads == 0, "attention_hidden_dim must be divisible by n_heads"
        self.head_dim = attention_hidden_dim // n_heads
        self.query_proj = nn.Linear(query_input_dim, attention_hidden_dim, bias=False)
        self.key_proj = nn.Linear(context_node_input_dim, attention_hidden_dim, bias=False)
        self.norm_factor = 1 / math.sqrt(self.head_dim)
        self.final_fc_for_probs = nn.Linear(attention_hidden_dim, 1, bias=False)
        if INIT:
            nn.init.xavier_uniform_(self.query_proj.weight, gain=1.414)
            nn.init.xavier_uniform_(self.key_proj.weight, gain=1.414)
            nn.init.xavier_uniform_(self.final_fc_for_probs.weight, gain=1.414)

    def forward(self, current_decoder_state_query, context_nodes_for_kv, mask):
        batch_size, n_nodes, _ = context_nodes_for_kv.size()
        Q_simple = self.query_proj(current_decoder_state_query).unsqueeze(1)
        K_simple = self.key_proj(context_nodes_for_kv)
        compatibility_simple = torch.bmm(Q_simple, K_simple.transpose(1, 2)) * (1 / math.sqrt(K_simple.size(-1)))
        compatibility_simple = compatibility_simple.squeeze(1)
        compatibility_simple = torch.tanh(compatibility_simple) * 10.0
        # print(f"DEBUG PA: Q_simple NaNs: {torch.isnan(Q_simple).any()}")
        # print(f"DEBUG PA: K_simple NaNs: {torch.isnan(K_simple).any()}")

        compatibility_simple = torch.bmm(Q_simple, K_simple.transpose(1, 2)) * (1 / math.sqrt(K_simple.size(-1)))
        # print(f"DEBUG PA: compatibility_simple after bmm NaNs: {torch.isnan(compatibility_simple).any()}")

        compatibility_simple = torch.tanh(compatibility_simple) * 10.0
        # print(f"DEBUG PA: compatibility_simple after tanh*10 NaNs: {torch.isnan(compatibility_simple).any()}")
        if mask is not None:
            # Make sure mask is float for multiplication if using it that way, or bool for masked_fill
            # The current code uses masked_fill, which is fine.
            # Check if the mask is causing all values to be -inf
            # print(f"DEBUG PA: Mask shape: {mask.shape}, Mask True count: {mask.sum()}")
            _temp_compat = compatibility_simple.masked_fill(mask.bool(), float("-inf"))
            # print(f"DEBUG PA: _temp_compat all -inf: {(torch.isinf(_temp_compat) & (_temp_compat < 0)).all()}")

        compatibility_simple = compatibility_simple.masked_fill(mask.bool(), float("-inf"))
        scores = F.softmax(compatibility_simple, dim=-1)
        # print(f"DEBUG PA: scores (action_probs) NaNs: {torch.isnan(scores).any()}")
        return scores


class Decoder1(nn.Module):
    def __init__(self, node_embedding_dim, decoder_hidden_dim, prob_attention_heads=8):
        super(Decoder1, self).__init__()
        self.node_embedding_dim = node_embedding_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.greedy_eval_mode = False  # MODIFIED: Initialize greedy_eval_mode

        self.fc_state_input_proj = nn.Linear(node_embedding_dim + 1, decoder_hidden_dim, bias=False)
        self.fc_pool_proj = nn.Linear(node_embedding_dim, decoder_hidden_dim, bias=False)
        context_node_dim = self.node_embedding_dim + 1
        self.prob_attention = ProbAttention(
            n_heads=prob_attention_heads,
            query_input_dim=decoder_hidden_dim,
            context_node_input_dim=context_node_dim,  # <-- The dimension is now larger
            attention_hidden_dim=decoder_hidden_dim
        )
        if INIT:
            nn.init.xavier_uniform_(self.fc_state_input_proj.weight, gain=1.414)
            nn.init.xavier_uniform_(self.fc_pool_proj.weight, gain=1.414)

    def forward(self, current_batch_data: Batch,
                encoder_node_embeddings_flat: torch.Tensor,
                pooled_graph_embedding_batched: torch.Tensor,
                old_actions_from_memory: torch.Tensor = None,
                n_total_steps_max: int = 50,
                is_eval_pass: bool = False
                ):
        batch_size = current_batch_data.num_graphs
        if batch_size > 0 and hasattr(current_batch_data, 'ptr'):
            nodes_per_sample_counts = current_batch_data.ptr[1:] - current_batch_data.ptr[:-1]
            if not torch.all(nodes_per_sample_counts == nodes_per_sample_counts[0]):
                raise ValueError("Decoder currently assumes all graphs in a batch have the same number of nodes.")
            n_nodes_per_sample = nodes_per_sample_counts[0].item()
        elif batch_size == 1:
            n_nodes_per_sample = current_batch_data.num_nodes
        else:
            if encoder_node_embeddings_flat.numel() == 0:
                if is_eval_pass:
                    return torch.empty(0, device=device), torch.empty(0, device=device)
                else:
                    return torch.empty(0, 0, device=device, dtype=torch.long), \
                        torch.empty(0, device=device), \
                        torch.empty(0, device=device)
            raise ValueError(
                f"Cannot determine n_nodes_per_sample for batch_size {batch_size} and embedding size {encoder_node_embeddings_flat.shape}")

        encoder_nodes_batched = encoder_node_embeddings_flat.reshape(batch_size, n_nodes_per_sample,
                                                                     self.node_embedding_dim)
        demands_batched = current_batch_data.demand.reshape(batch_size, n_nodes_per_sample)
        static_vehicle_capacity_tensor = current_batch_data.capacity
        dynamic_vehicle_capacity = static_vehicle_capacity_tensor.clone()

        scalar_static_capacity = static_vehicle_capacity_tensor[0, 0].item() if batch_size > 0 else 0.0
        depot_node_idx_local = n_nodes_per_sample - 1

        all_due_dates_batched = current_batch_data.x[:, 4].reshape(batch_size, n_nodes_per_sample)
        batch_current_subtour_start_date = torch.full((batch_size,), -1.0, dtype=torch.float, device=device)
        dynamic_due_dates_batched = all_due_dates_batched.clone()
        all_intervals_batched = current_batch_data.x[:, 1].clone().reshape(batch_size, n_nodes_per_sample)

        visited_mask_permanent = torch.zeros((batch_size, n_nodes_per_sample), dtype=torch.bool, device=device)
        current_selected_node_idx_local = torch.full((batch_size,), depot_node_idx_local, dtype=torch.long,
                                                     device=device)
        log_probs_trajectory, actions_trajectory, entropies_trajectory = [], [], []

        batch_current_subtour_skills = [set() for _ in range(batch_size)]
        batch_current_subtour_panels = [set() for _ in range(batch_size)]
        batch_current_subtour_nodes_indices_local = [[] for _ in range(batch_size)]
        batch_current_subtour_labour_sum = [0.0] * batch_size
        batch_all_completed_subtour_labours = [[] for _ in range(batch_size)]
        total_rewards_for_batch = torch.zeros(batch_size, device=device)
        current_node_features_batched = encoder_nodes_batched.gather(
            1, current_selected_node_idx_local.view(-1, 1, 1).expand(-1, -1, self.node_embedding_dim)
        ).squeeze(1)

        current_step_mask, visited_mask_permanent = update_mask(
            demands_batched, dynamic_vehicle_capacity,
            current_selected_node_idx_local.unsqueeze(-1),
            visited_mask_permanent,
            depot_node_idx_local,
            all_due_dates_batched,  # New argument
            batch_current_subtour_start_date  # New argument
        )
        processed_pooled_embed = self.fc_pool_proj(pooled_graph_embedding_batched)

        for step_t in range(n_total_steps_max):
            state_input_part = self.fc_state_input_proj(
                torch.cat([current_node_features_batched, dynamic_vehicle_capacity], dim=-1))
            decoder_query = state_input_part + processed_pooled_embed
            # print(f"DEBUG: Decoder query contains NaNs: {torch.isnan(decoder_query).any()}")
            # print(f"DEBUG: Encoder nodes (batched) contains NaNs: {torch.isnan(encoder_nodes_batched).any()}")
            # print(f"DEBUG: current_step_mask (step {step_t}): {current_step_mask}")
            # print(f"DEBUG: Are all actions masked for any batch item? {current_step_mask.all(dim=1).any()}")
            dynamic_context_nodes = torch.cat(
                [encoder_nodes_batched, demands_batched.unsqueeze(-1)],
                dim=-1
            )

            # Now, pass this new dynamic context to the attention module
            action_probs = self.prob_attention(decoder_query, dynamic_context_nodes, current_step_mask)

            # if not is_eval_pass and current_step_mask[:, :depot_node_idx_local].all(dim=1).any():
            #     # Check if any item in the batch has all its task nodes masked
            #     finished_mask = current_step_mask[:, :depot_node_idx_local].all(dim=1)
            #     if finished_mask.any():
            #         print(
            #             f"DEBUG: Step {step_t}, Batch items {torch.where(finished_mask)[0].tolist()} are functionally complete. All tasks masked.")
            # print(f"DEBUG: action_probs after ProbAttention: {action_probs}")  # Check this too
            # print(f"DEBUG: action_probs contains NaNs: {torch.isnan(action_probs).any()}")
            if action_probs.ndim == 3:
                # Assuming dim 1 is the 'n_heads' dimension
                # Average probabilities across the heads
                action_probs = action_probs.mean(dim=1)
                # print(f"DEBUG Decoder1.forward: action_probs.shape AFTER averaging heads: {action_probs.shape}")
            elif action_probs.ndim == 2:
                # It's already [batch_size, num_nodes], which is expected if ProbAttention is single-head
                print(f"DEBUG Decoder1.forward: action_probs is already 2D: {action_probs.shape}")
            else:
                # Handle unexpected shape
                raise ValueError(
                    f"Unexpected shape for action_probs: {action_probs.shape}. Expected 2D or 3D (with head dim).")
            # --- NEW MODIFICATION END ---

            dist = Categorical(probs=action_probs)

            if is_eval_pass:
                # ... (rest of is_eval_pass = True logic)
                # current_action_local will be defined from old_actions_from_memory
                # Ensure loop breaks if old_actions_from_memory runs out for step_t
                if old_actions_from_memory is None:  # From original file
                    break
                if step_t >= old_actions_from_memory.size(1):
                    # print(
                    #     f"DEBUG Decoder1.forward (eval): step_t {step_t} exceeds old_actions_from_memory length {old_actions_from_memory.size(1)}. Breaking.")
                    break
                current_action_local = old_actions_from_memory[:, step_t]

                log_p, entropy = dist.log_prob(current_action_local), dist.entropy()  #
                log_probs_trajectory.append(log_p.unsqueeze(1));
                entropies_trajectory.append(entropy.unsqueeze(1))
            else:
                current_action_local = torch.argmax(action_probs, dim=1) if self.greedy_eval_mode else dist.sample()  #
                # Add debug print for current_action_local's shape here if needed:
                # print(f"DEBUG Decoder1.forward (act): current_action_local.shape: {current_action_local.shape}")

                log_p = dist.log_prob(current_action_local)  #
                actions_trajectory.append(current_action_local.unsqueeze(1));  #
                log_probs_trajectory.append(log_p.unsqueeze(1))  #

                current_step_batch_rewards = torch.zeros(batch_size, device=device)  #

                # Note: You have a nested 'for b_idx in range(batch_size):' loop here.
                # The outer one seems correct for iterating through batch items for rewards.
                # The inner one is redundant and likely a copy-paste error. I'm removing the inner one.
                for b_idx in range(batch_size):
                    selected_node_local_idx = current_action_local[b_idx].item()

                    # --- NEW DEPOT REWARD LOGIC ---
                    if selected_node_local_idx == depot_node_idx_local:
                        # Get the list of tasks from the subtour that just ended
                        tasks_in_completed_subtour = batch_current_subtour_nodes_indices_local[b_idx]
                        num_tasks_in_subtour = len(tasks_in_completed_subtour)

                        if num_tasks_in_subtour > 0:
                            # --- Dynamic Subtour Completion Bonus ---
                            # The reward is proportional to the number of tasks completed.
                            # This incentivizes longer, more productive subtours.
                            # 'bonus_per_task' is a new hyperparameter you can tune.
                            bonus_per_task = 0.5
                            completion_bonus = num_tasks_in_subtour * bonus_per_task
                            reward_for_this_task = completion_bonus
                        else:
                            # Apply a penalty for empty, wasted trips to the depot
                            reward_for_this_task = -0.5

                        current_step_batch_rewards[b_idx] += reward_for_this_task

                        # This part remains the same: reset state for the next subtour
                        if batch_current_subtour_labour_sum[b_idx] > 0:
                            batch_all_completed_subtour_labours[b_idx].append(batch_current_subtour_labour_sum[b_idx])

                        batch_current_subtour_start_date[b_idx] = -1.0
                        batch_current_subtour_skills[b_idx].clear()
                        batch_current_subtour_panels[b_idx].clear()
                        batch_current_subtour_nodes_indices_local[b_idx] = []
                        batch_current_subtour_labour_sum[b_idx] = 0.0

                    # --- TASK REWARD LOGIC (Remains the same as your code) ---
                    else:
                        global_node_idx = current_batch_data.ptr[b_idx] + selected_node_local_idx
                        raw_node_features_selected = current_batch_data.x[global_node_idx]
                        panel_info_selected = current_batch_data.raw_panel[b_idx][selected_node_local_idx]

                        task_duedate = dynamic_due_dates_batched[b_idx, selected_node_local_idx].item()

                        # Get the other, static features from the raw data as before
                        task_labour, task_skill, task_interval = raw_node_features_selected[2].item(), \
                            raw_node_features_selected[3].item(), raw_node_features_selected[1].item()

                        batch_current_subtour_labour_sum[b_idx] += task_labour
                        reward_for_this_task = 0.0

                        if batch_current_subtour_start_date[b_idx] == -1.0:
                            batch_current_subtour_start_date[b_idx] = task_duedate

                        date_diff = abs(task_duedate - batch_current_subtour_start_date[b_idx])
                        threshold = task_interval * 0.50

                        if date_diff <= threshold:
                            date_diff_norm_local = date_diff / (threshold + 1e-6)
                            reward_for_this_task += (1.0 - date_diff_norm_local) * 1.5
                            # print(
                            #     f"Step Reward: POSITIVE (+{reward_for_this_task:.2f}) for task {selected_node_local_idx} with date_diff {date_diff:.1f}")
                        else:
                            overage_norm = (date_diff - threshold) / (task_interval + 1e-6)
                            distant_due_date_penalty = -overage_norm * 0.2

                            # Clip the penalty to a maximum negative value (e.g., -2.0)
                            # You can tune this maximum penalty value.
                            clipped_penalty = max(distant_due_date_penalty, -1.0)

                            reward_for_this_task += clipped_penalty
                            # print(
                            #     f"Step Reward: PENALTY ({clipped_penalty:.2f}) for task {selected_node_local_idx} with date_diff {date_diff:.1f}")

                        if task_skill in batch_current_subtour_skills[b_idx]:
                            reward_for_this_task += 0.2
                            # print("Step Reward: Skill bonus +0.2")
                        batch_current_subtour_skills[b_idx].add(task_skill)

                        individual_panels_for_current_task = []
                        if isinstance(panel_info_selected, str) and \
                                panel_info_selected != "0" and \
                                panel_info_selected.upper() != "NOTE" and \
                                panel_info_selected.upper() != "DEPOT":
                            potential_panels = [p.strip() for p in panel_info_selected.split(',')]
                            individual_panels_for_current_task = [p for p in potential_panels if
                                                                  p and p != "0" and p.upper() != "NOTE"]
                        if individual_panels_for_current_task:
                            found_reuse_for_this_task = False
                            for single_panel_code in individual_panels_for_current_task:
                                if single_panel_code in batch_current_subtour_panels[b_idx]:
                                    found_reuse_for_this_task = True
                                    break
                            if found_reuse_for_this_task:
                                reward_for_this_task += 0.3
                                # print("Step Reward: Panel bonus +0.3")
                            for single_panel_code in individual_panels_for_current_task:
                                batch_current_subtour_panels[b_idx].add(single_panel_code)

                        batch_current_subtour_nodes_indices_local[b_idx].append(selected_node_local_idx)
                        current_step_batch_rewards[b_idx] += reward_for_this_task
                        # print(f"Depot Reward: {reward_for_this_task:.2f}")

                # This line should be outside your loop
                total_rewards_for_batch += current_step_batch_rewards

                # This check is crucial if the loop could have broken early in the is_eval_pass branch
            if 'current_action_local' not in locals() and step_t < n_total_steps_max:
                # This implies the loop broke, likely from the is_eval_pass branch due to exhausting old_actions_from_memory
                # No further state updates should happen for this step if actions weren't determined.
                print(
                    f"DEBUG Decoder1.forward: Loop broke at step_t {step_t} before current_action_local was finalized for state update. Ending episode pass.")
                break  # Exit the main step_t loop



            # Update the full STATE, including the dynamic due dates
            dynamic_vehicle_capacity, demands_batched, dynamic_due_dates_batched = update_state(
                demands_batched,
                dynamic_vehicle_capacity,
                dynamic_due_dates_batched,  # <-- Pass in current due dates
                all_intervals_batched,  # <-- Pass in intervals
                batch_current_subtour_start_date,  # <-- Pass in the completion date
                current_action_local.unsqueeze(-1),
                scalar_static_capacity,
                depot_node_idx_local
            )

            current_step_mask, visited_mask_permanent = update_mask(
                demands_batched, dynamic_vehicle_capacity,
                current_action_local.unsqueeze(-1),
                visited_mask_permanent,
                depot_node_idx_local,
                dynamic_due_dates_batched,  # <-- Use the dynamic tensor here
                batch_current_subtour_start_date
            )
            current_selected_node_idx_local = current_action_local
            current_node_features_batched = encoder_nodes_batched.gather(1, current_selected_node_idx_local.view(-1, 1,
                                                                                                                 1).expand(
                -1, -1, self.node_embedding_dim)).squeeze(1)

            all_demand_met = demands_batched[:, :depot_node_idx_local].le(0).all()

            if all_demand_met:
                # If and only if all task demands are met, the episode is truly over.
                break

        if not is_eval_pass:
            # --- START: PROPORTIONAL GLOBAL COMPLETION REWARD (Corrected) ---

            # 1. Calculate the completion ratio for each item in the batch

            # CORRECTED: Use global_add_pool to sum initial demands for EACH graph.
            # This creates a tensor of shape [batch_size].
            initial_demand_per_graph = global_add_pool(current_batch_data.demand, current_batch_data.batch).squeeze(-1)

            # This calculation was already correct, with shape [batch_size].
            remaining_demand_per_graph = demands_batched.sum(dim=1)

            # Avoid division by zero
            initial_demand_per_graph[initial_demand_per_graph == 0] = 1.0

            # NOW THE SHAPES MATCH: completion_ratio will have shape [batch_size]
            completion_ratio = 1.0 - (remaining_demand_per_graph / initial_demand_per_graph)

            # 2. Initialize a tensor to hold the new rewards/penalties
            terminal_rewards = torch.zeros_like(total_rewards_for_batch)

            # --- The rest of the tiered reward logic is correct and does not need to change ---
            # --- Tier 1: High Completion Bonus (> 95%) ---
            high_completion_mask = completion_ratio > 0.95
            terminal_rewards[high_completion_mask] = 20

            # --- Tier 2: Mid-Tier Proportional Reward (50% to 95%) ---
            mid_completion_mask = (completion_ratio >= 0.50) & (completion_ratio <= 0.95)
            mid_rewards = ((completion_ratio[mid_completion_mask] - 0.50) / (0.95 - 0.50)) * 20
            terminal_rewards[mid_completion_mask] = mid_rewards

            # --- Tier 3: Low Completion Penalty (< 50%) ---
            low_completion_mask = completion_ratio < 0.50
            low_rewards_penalty = (completion_ratio[low_completion_mask] - 0.50) * 10
            terminal_rewards[low_completion_mask] = low_rewards_penalty

            # 3. Add the calculated terminal rewards to the total rewards for the batch
            total_rewards_for_batch += terminal_rewards

            # --- END: PROPORTIONAL GLOBAL COMPLETION REWARD ---
            labour_balance_penalty = 0.0
            # This is your existing labor variance penalty code
            for b_idx in range(batch_size):
                subtour_labours = batch_all_completed_subtour_labours[b_idx]
                if len(subtour_labours) > 1:
                    avg_labour = sum(subtour_labours) / len(subtour_labours)
                    variance = sum((x - avg_labour) ** 2 for x in subtour_labours) / len(subtour_labours)
                    labour_balance_penalty = variance / (avg_labour ** 2 + 1e-6)
                    total_rewards_for_batch[b_idx] -= labour_balance_penalty * 0.1

            # print(f"--- Episode End ---")
            # print(
            #     f"[DEBUG] Initial Demand: {initial_demand_per_graph.mean().item():.2f}, Remaining Demand: {remaining_demand_per_graph.mean().item():.2f}, Completion Ratio: {completion_ratio.mean().item():.2f}")
            # print(f"Completion Ratio: {completion_ratio.mean().item():.2f}, Terminal Reward: {terminal_rewards.mean().item():.2f}")
            # print(f"Labor Balance Penalty: {-labour_balance_penalty * 0.1:.2f}") # Assuming you have this calculated per-item
            # print(f"Final Total Reward: {total_rewards_for_batch.mean().item():.2f}")

        if is_eval_pass:
            entropies_cat = torch.cat(entropies_trajectory, dim=1) if entropies_trajectory else torch.empty(batch_size,
                                                                                                            0,
                                                                                                            device=device)
            old_log_probs_cat = torch.cat(log_probs_trajectory, dim=1) if log_probs_trajectory else torch.empty(
                batch_size, 0, device=device)
            mean_entropies = entropies_cat.mean(dim=1) if entropies_cat.numel() > 0 else torch.zeros(batch_size,
                                                                                                     device=device)
            return mean_entropies, old_log_probs_cat.sum(dim=1)
        else:
            actions_cat = torch.cat(actions_trajectory, dim=1) if actions_trajectory else torch.empty(batch_size, 0,
                                                                                                      dtype=torch.long,
                                                                                                      device=device)
            log_probs_cat = torch.cat(log_probs_trajectory, dim=1) if log_probs_trajectory else torch.empty(batch_size,
                                                                                                            0,
                                                                                                            device=device)
            return actions_cat, log_probs_cat.sum(dim=1), total_rewards_for_batch


class Model(nn.Module):
    def __init__(self, raw_node_feature_dim, demand_feature_dim, hidden_node_dim,
                 input_edge_dim, hidden_edge_dim, conv_layers,
                 decoder_hidden_dim, prob_attention_heads):
        super(Model, self).__init__()
        self.encoder = Encoder(raw_node_feature_dim, demand_feature_dim, hidden_node_dim,
                               input_edge_dim, hidden_edge_dim, conv_layers)
        self.decoder = Decoder1(hidden_node_dim, decoder_hidden_dim, prob_attention_heads)

    def forward(self, current_batch_data: Batch,
                old_actions_from_memory: torch.Tensor = None,
                n_total_steps_max: int = 50,
                is_eval_pass: bool = False):
        encoder_node_embeddings_flat = self.encoder(current_batch_data)
        batch_size = current_batch_data.num_graphs
        if batch_size > 0:
            # MODIFIED: Using global_mean_pool
            pooled_graph_embedding_batched = global_mean_pool(encoder_node_embeddings_flat, current_batch_data.batch)
        elif encoder_node_embeddings_flat.numel() == 0:
            pooled_graph_embedding_batched = torch.empty(0, encoder_node_embeddings_flat.size(-1),
                                                         device=encoder_node_embeddings_flat.device)
        else:
            raise ValueError(
                f"Batch size {batch_size} and embedding shape {encoder_node_embeddings_flat.shape} mismatch for pooling.")

        if is_eval_pass:
            mean_entropies, sum_old_log_probs = self.decoder(
                current_batch_data, encoder_node_embeddings_flat, pooled_graph_embedding_batched,
                old_actions_from_memory, n_total_steps_max, is_eval_pass=True
            )
            return mean_entropies, sum_old_log_probs, encoder_node_embeddings_flat
        else:
            actions, log_probs, rewards = self.decoder(
                current_batch_data, encoder_node_embeddings_flat, pooled_graph_embedding_batched,
                old_actions_from_memory=None,
                n_total_steps_max=n_total_steps_max, is_eval_pass=False
            )
            return actions, log_probs, rewards, encoder_node_embeddings_flat


class Critic(nn.Module):
    def __init__(self, hidden_node_dim_from_encoder):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(hidden_node_dim_from_encoder, hidden_node_dim_from_encoder // 2)
        self.fc2 = nn.Linear(hidden_node_dim_from_encoder // 2, 1)
        if INIT:
            nn.init.orthogonal_(self.fc1.weight, gain=math.sqrt(2))
            if self.fc1.bias is not None: nn.init.zeros_(self.fc1.bias)
            nn.init.orthogonal_(self.fc2.weight, gain=1.0)
            if self.fc2.bias is not None: nn.init.zeros_(self.fc2.bias)

    def forward(self, encoder_node_embeddings_flat, batch_vector):
        if encoder_node_embeddings_flat.numel() == 0:
            return torch.empty(0, device=encoder_node_embeddings_flat.device)
        pooled_graph_features = global_mean_pool(encoder_node_embeddings_flat, batch_vector)
        value = F.relu(self.fc1(pooled_graph_features))
        value = self.fc2(value)
        return value.squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, raw_node_feature_dim, demand_feature_dim, hidden_node_dim,
                 input_edge_dim, hidden_edge_dim, conv_layers,
                 decoder_hidden_dim, prob_attention_heads):
        super(ActorCritic, self).__init__()
        self.actor = Model(raw_node_feature_dim, demand_feature_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim,
                           conv_layers, decoder_hidden_dim, prob_attention_heads)
        self.critic = Critic(hidden_node_dim)

    def act(self, current_batch_data: Batch, actions_placeholder, n_total_steps_max: int, greedy: bool,
            is_eval_pass: bool):
        if hasattr(self.actor.decoder, 'greedy_eval_mode'): self.actor.decoder.greedy_eval_mode = greedy
        # MODIFIED: is_eval_pass is hardcoded to False for self.actor call during .act()
        actions, log_probs, rewards, _ = self.actor(current_batch_data, None, n_total_steps_max, is_eval_pass=False)
        return actions, log_probs, rewards

    def evaluate(self, current_batch_data: Batch, actions_from_memory: torch.Tensor, n_total_steps_max: int):
        mean_entropies, sum_old_log_probs, encoder_node_embeddings_flat = self.actor(current_batch_data,
                                                                                     actions_from_memory,
                                                                                     n_total_steps_max,
                                                                                     is_eval_pass=True)
        state_values = self.critic(encoder_node_embeddings_flat, current_batch_data.batch)
        return mean_entropies, sum_old_log_probs, state_values


class Memory:
    def __init__(self): self.batches_data, self.actions, self.log_probs, self.rewards = [], [], [], []

    def clear(self): del self.batches_data[:]; del self.actions[:]; del self.log_probs[:]; del self.rewards[:]

    def store(self, batch_data, action, log_prob, reward):
        self.batches_data.append(batch_data.cpu());
        self.actions.append(action.cpu());
        self.log_probs.append(log_prob.cpu());
        self.rewards.append(reward.cpu())

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class AgentPPO:
    def __init__(self, raw_node_feature_dim, demand_feature_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim,
                 conv_layers, decoder_hidden_dim, prob_attention_heads, lr, ppo_epochs, ppo_clip_eps, entropy_coeff,
                 batch_size_ppo_update, total_epochs, data_loader_len):

        # Create the base models first
        self.policy = ActorCritic(raw_node_feature_dim, demand_feature_dim, hidden_node_dim, input_edge_dim,
                                  hidden_edge_dim, conv_layers, decoder_hidden_dim, prob_attention_heads)
        self.old_policy = ActorCritic(raw_node_feature_dim, demand_feature_dim, hidden_node_dim, input_edge_dim,
                                      hidden_edge_dim, conv_layers, decoder_hidden_dim, prob_attention_heads)

        # Check for multiple GPUs and wrap with DataParallel if available
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for parallel processing.")
            self.policy = nn.DataParallel(self.policy)
            self.old_policy = nn.DataParallel(self.old_policy)

        # Move models to the primary device
        self.policy.to(device)
        self.old_policy.to(device)

        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        # --- START OF CHANGE ---
        # Replace the StepLR with a more robust warm-up and decay scheduler.

        # Calculate total training steps
        num_training_steps = total_epochs * data_loader_len
        # Set a number of steps for warm-up (e.g., 5% of total steps)
        num_warmup_steps = int(num_training_steps * 0.05)

        print(f"LR Scheduler: {num_training_steps} total steps, {num_warmup_steps} warmup steps.")

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        # --- END OF CHANGE ---
        self.mse_loss = nn.MSELoss()
        self.ppo_epochs, self.ppo_clip_eps, self.entropy_coeff, self.batch_size_ppo_update = ppo_epochs, ppo_clip_eps, entropy_coeff, batch_size_ppo_update

    # def update(self, memory: Memory, current_training_epoch: int):
    #     # MODIFIED: PPOUpdateDataset and ppo_collate_fn_corrected for correct trajectory batching
    #     class PPOUpdateDataset(torch.utils.data.Dataset):
    #         def __init__(self, memory_batches_in, memory_actions_in, memory_log_probs_in, memory_rewards_in):
    #             self.all_actions = torch.cat(memory_actions_in, dim=0)
    #             self.all_log_probs = torch.cat(memory_log_probs_in, dim=0)
    #             self.all_rewards = torch.cat(memory_rewards_in, dim=0)
    #             self.all_graphs = []
    #             for pyg_batch in memory_batches_in:
    #                 for i in range(pyg_batch.num_graphs):
    #                     self.all_graphs.append(pyg_batch[i])
    #
    #             num_total_trajectories = self.all_actions.size(0)
    #             if not (len(self.all_graphs) == num_total_trajectories and \
    #                     self.all_log_probs.size(0) == num_total_trajectories and \
    #                     self.all_rewards.size(0) == num_total_trajectories):
    #                 raise ValueError(f"Mismatch in trajectory component lengths after processing memory: "
    #                                  f"Graphs: {len(self.all_graphs)}, Actions: {self.all_actions.size(0)}, "
    #                                  f"LogProbs: {self.all_log_probs.size(0)}, Rewards: {self.all_rewards.size(0)}")
    #
    #         def __len__(self):
    #             return self.all_actions.size(0)
    #
    #         def __getitem__(self, idx):
    #             return (self.all_graphs[idx], self.all_actions[idx], self.all_log_probs[idx], self.all_rewards[idx])
    #
    #     def ppo_collate_fn_corrected(samples):
    #         batch_data_list, actions_list, log_probs_list, rewards_list = zip(*samples)
    #         collated_batch_data = Batch.from_data_list(list(batch_data_list)).to(device)
    #         collated_actions = torch.stack(list(actions_list)).to(device)
    #         collated_log_probs = torch.stack(list(log_probs_list)).to(device)
    #         collated_rewards = torch.stack(list(rewards_list)).to(device)
    #         return collated_batch_data, collated_actions, collated_log_probs, collated_rewards
    #
    #     if not memory.actions or not memory.log_probs or not memory.rewards or not memory.batches_data:
    #         return
    #
    #     ppo_dataset = PPOUpdateDataset(memory.batches_data, memory.actions, memory.log_probs, memory.rewards)
    #     if len(ppo_dataset) == 0: return
    #
    #     ppo_dataloader = torch.utils.data.DataLoader(
    #         ppo_dataset, batch_size=self.batch_size_ppo_update, shuffle=True, collate_fn=ppo_collate_fn_corrected
    #     )
    #
    #     for _ in range(self.ppo_epochs):
    #         if len(ppo_dataloader) == 0: break
    #         for b_data_traj, b_actions_traj, b_old_log_probs_traj, b_rewards_traj in ppo_dataloader:
    #             if b_actions_traj.size(0) == 0: continue
    #             trajectory_len = b_actions_traj.size(1)
    #             if trajectory_len == 0: continue
    #
    #             entropies, log_probs, state_values = self.policy.evaluate(b_data_traj, b_actions_traj, trajectory_len)
    #             advantages = b_rewards_traj - state_values.detach()
    #             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #             ratios = torch.exp(log_probs - b_old_log_probs_traj.detach())
    #             surr1, surr2 = ratios * advantages, torch.clamp(ratios, 1.0 - self.ppo_clip_eps,
    #                                                             1.0 + self.ppo_clip_eps) * advantages
    #             actor_loss = -torch.min(surr1, surr2).mean()
    #             critic_loss = self.mse_loss(state_values, b_rewards_traj)
    #             entropy_loss = -self.entropy_coeff * entropies.mean() if entropies.numel() > 0 else torch.tensor(0.0,
    #                                                                                                              device=device)
    #             total_loss = actor_loss + 0.5 * critic_loss + entropy_loss
    #             self.optimizer.zero_grad();
    #             total_loss.backward();
    #             torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm);
    #             self.optimizer.step()
    #     self.old_policy.load_state_dict(self.policy.state_dict())
    def update(self, memory: Memory, current_training_epoch: int):
        # print("DEBUG: Entering AgentPPO.update()")  # Added
        all_actor_losses_for_update = []
        all_critic_losses_for_update = []
        all_entropy_losses_for_update = []
        # MODIFIED: PPOUpdateDataset and ppo_collate_fn_corrected for correct trajectory batching
        class PPOUpdateDataset(torch.utils.data.Dataset):
            def __init__(self, memory_batches_in, memory_actions_in, memory_log_probs_in, memory_rewards_in):
                max_len = max([t.size(1) for t in memory_actions_in])

                # 2. Pad ONLY the actions tensor
                padded_actions = []
                for actions in memory_actions_in:
                    # Pad the actions tensor to the max length
                    padded_a = torch.nn.functional.pad(actions, (0, max_len - actions.size(1)), 'constant', 0)
                    padded_actions.append(padded_a)

                # 3. Concatenate the padded actions
                self.all_actions = torch.cat(padded_actions, dim=0)

                # 4. Concatenate log_probs and rewards directly, as they do not need padding
                self.all_log_probs = torch.cat(memory_log_probs_in, dim=0)
                self.all_rewards = torch.cat(memory_rewards_in, dim=0)
                self.all_graphs = []
                for pyg_batch in memory_batches_in:
                    for i in range(pyg_batch.num_graphs):
                        self.all_graphs.append(pyg_batch[i])

                num_total_trajectories = self.all_actions.size(0)
                if not (len(self.all_graphs) == num_total_trajectories and \
                        self.all_log_probs.size(0) == num_total_trajectories and \
                        self.all_rewards.size(0) == num_total_trajectories):
                    print(f"DEBUG: Mismatch in trajectory component lengths! "  # Added
                          f"Graphs: {len(self.all_graphs)}, Actions: {self.all_actions.size(0)}, "
                          f"LogProbs: {self.all_log_probs.size(0)}, Rewards: {self.all_rewards.size(0)}")
                    # Consider raising an error here if this happens
                    # raise ValueError("Mismatch in trajectory component lengths")
                # print(f"DEBUG: PPOUpdateDataset created. Total trajectories: {len(self.all_graphs)}")  # Added

            def __len__(self):
                return self.all_actions.size(0)

            def __getitem__(self, idx):
                return (self.all_graphs[idx], self.all_actions[idx], self.all_log_probs[idx], self.all_rewards[idx])

        def ppo_collate_fn_corrected(samples):
            # ... (no changes needed here for now)
            batch_data_list, actions_list, log_probs_list, rewards_list = zip(*samples)
            collated_batch_data = Batch.from_data_list(list(batch_data_list)).to(device)
            collated_actions = torch.stack(list(actions_list)).to(device)
            collated_log_probs = torch.stack(list(log_probs_list)).to(device)
            collated_rewards = torch.stack(list(rewards_list)).to(device)
            return collated_batch_data, collated_actions, collated_log_probs, collated_rewards

        if not memory.actions or not memory.log_probs or not memory.rewards or not memory.batches_data:
            print("DEBUG: AgentPPO.update() called with empty memory. Skipping update.")  # Added
            return None, None, None

        # print("DEBUG: Creating PPOUpdateDataset...")  # Added
        ppo_dataset = PPOUpdateDataset(memory.batches_data, memory.actions, memory.log_probs, memory.rewards)

        if len(ppo_dataset) == 0:
            print("DEBUG: AgentPPO.update() ppo_dataset is empty. Skipping update.")  # Added
            return None, None, None
        # print(f"DEBUG: PPOUpdateDataset length: {len(ppo_dataset)}")  # Added

        # print("DEBUG: Creating ppo_dataloader...")  # Added
        ppo_dataloader = torch.utils.data.DataLoader(
            ppo_dataset, batch_size=self.batch_size_ppo_update, shuffle=True, collate_fn=ppo_collate_fn_corrected
        )
        # print(f"DEBUG: ppo_dataloader created. Number of PPO minibatches: {len(ppo_dataloader)}")  # Added
        accumulation_steps = 8
        try:  # Added try-except block
            for ppo_ep in range(self.ppo_epochs):
                # print(f"DEBUG: Starting PPO Epoch {ppo_ep + 1}/{self.ppo_epochs}")  # Added
                if len(ppo_dataloader) == 0:
                    print("DEBUG: AgentPPO.update() ppo_dataloader is empty inside epoch loop. Breaking.")  # Added
                    break
                for batch_idx, (b_data_traj, b_actions_traj, b_old_log_probs_traj, b_rewards_traj) in enumerate(
                        ppo_dataloader):
                    print(f"DEBUG: PPO Epoch {ppo_ep + 1}, Minibatch {batch_idx + 1}/{len(ppo_dataloader)}")  # Added

                    if b_actions_traj.size(0) == 0:
                        print("DEBUG: Empty actions tensor in minibatch, skipping.")  # Added
                        continue
                    trajectory_len = b_actions_traj.size(1)
                    if trajectory_len == 0:
                        print("DEBUG: Zero trajectory length in minibatch, skipping.")  # Added
                        continue

                    # Policy evaluation (forward pass)
                    # print("DEBUG: Evaluating policy...")  # Added
                    policy_module = self.policy.module if isinstance(self.policy, nn.DataParallel) else self.policy
                    if isinstance(self.policy, nn.DataParallel):
                        # Access the original model using .module
                        entropies, log_probs, state_values = self.policy.module.evaluate(b_data_traj, b_actions_traj,
                                                                                         trajectory_len)
                    else:
                        # If not using DataParallel, call it directly
                        entropies, log_probs, state_values = self.policy.evaluate(b_data_traj, b_actions_traj,
                                                                                  trajectory_len)
                    # print("DEBUG: Policy evaluated.")  # Added

                    # # Loss calculations
                    # advantages = b_rewards_traj - state_values.detach()
                    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    # ratios = torch.exp(log_probs - b_old_log_probs_traj.detach())
                    # surr1, surr2 = ratios * advantages, torch.clamp(ratios, 1.0 - self.ppo_clip_eps,
                    #                                                 1.0 + self.ppo_clip_eps) * advantages
                    # actor_loss = -torch.min(surr1, surr2).mean()
                    # critic_loss = self.mse_loss(state_values, b_rewards_traj)

                    # 1. Normalize the rewards from memory to have a mean of 0 and a std of 1.
                    #    This creates a stable target for the critic.
                    rewards_normalized = (b_rewards_traj - b_rewards_traj.mean()) / (b_rewards_traj.std() + 1e-8)

                    # 2. Calculate advantage using the NORMALIZED rewards and the critic's value estimate.
                    #    This ensures the actor and critic are operating on the same value scale.
                    advantages = rewards_normalized - state_values.detach()

                    # NOTE: The separate advantage normalization that was here before is now redundant
                    # because the rewards are already normalized, so it has been removed.

                    ratios = torch.exp(log_probs - b_old_log_probs_traj.detach())
                    surr1, surr2 = ratios * advantages, torch.clamp(ratios, 1.0 - self.ppo_clip_eps,
                                                                    1.0 + self.ppo_clip_eps) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # 3. Calculate critic loss using the NORMALIZED rewards as the target.
                    #    This will prevent the loss from exploding.
                    critic_loss = self.mse_loss(state_values, rewards_normalized)

                    entropy_loss = -self.entropy_coeff * entropies.mean() if entropies.numel() > 0 else torch.tensor(
                        0.0, device=device)
                    total_loss = actor_loss + 0.5 * critic_loss + entropy_loss
                    # print(f"DEBUG: Minibatch {batch_idx + 1} - Total Loss: {total_loss.item()}")  # Added
                    # --- START OF GRADIENT ACCUMULATION LOGIC ---

                    # 1. Scale the loss by the number of accumulation steps
                    loss = total_loss / accumulation_steps

                    # 2. Perform the backward pass to calculate gradients for this micro-batch
                    loss.backward()

                    # 3. Step the optimizer only after accumulating gradients for N steps
                    if (batch_idx + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()  # Step scheduler with the optimizer
                        self.optimizer.zero_grad()  # Clear gradients for the next accumulation cycle

                    # --- END OF LOGIC ---

                    # Optimization step (backward pass and optimizer step)
                    # self.optimizer.zero_grad();
                    # # print("DEBUG: Calculating gradients (backward pass)...")  # Added
                    # total_loss.backward();
                    # # print("DEBUG: Gradients calculated.")  # Added
                    # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm);
                    # # print("DEBUG: Stepping optimizer...")  # Added
                    # self.optimizer.step()
                    # self.scheduler.step()
                    # print("DEBUG: Optimizer stepped.")  # Added
                    all_actor_losses_for_update.append(actor_loss.item())
                    all_critic_losses_for_update.append(critic_loss.item())
                    all_entropy_losses_for_update.append(entropy_loss.item())
                # print(f"DEBUG: PPO Epoch {ppo_ep + 1} completed.")  # Added

            # print("DEBUG: All PPO epochs completed. Updating old_policy...")  # Added
            self.old_policy.load_state_dict(self.policy.state_dict())
            avg_actor_loss = np.mean(all_actor_losses_for_update) if all_actor_losses_for_update else 0.0
            avg_critic_loss = np.mean(all_critic_losses_for_update) if all_critic_losses_for_update else 0.0
            avg_entropy_loss = np.mean(all_entropy_losses_for_update) if all_entropy_losses_for_update else 0.0

            return avg_actor_loss, avg_critic_loss, avg_entropy_loss
            # print("DEBUG: AgentPPO.update() completed successfully.")  # Added

        except Exception as e:  # Added except block
            print(f"FATAL ERROR during AgentPPO.update() loop: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
            # You might want to re-raise the exception or handle it appropriately
            # raise e
