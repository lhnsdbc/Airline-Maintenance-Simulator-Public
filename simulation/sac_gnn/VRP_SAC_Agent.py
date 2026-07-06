# VRP_SAC_Agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
import numpy as np
import math
import random
from collections import deque

# Re-using the GNN Encoder from your original file
from .VRP_PPO_Model_revised import Encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer:
    """A simple FIFO experience replay buffer for SAC."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(
            *random.sample(self.buffer, batch_size))

        # Collate PyG Data objects into a single Batch object
        state_batch = Batch.from_data_list(state_batch).to(device)
        next_state_batch = Batch.from_data_list(next_state_batch).to(device)

        action_batch = torch.tensor(action_batch, dtype=torch.long, device=device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float, device=device).unsqueeze(1)
        done_batch = torch.tensor(done_batch, dtype=torch.float, device=device).unsqueeze(1)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """
    The SAC Actor (Policy) network.
    It takes node embeddings and produces action probabilities.
    This replaces the PPO Model/Decoder for action selection.
    """

    def __init__(self, hidden_node_dim, n_heads, n_nodes, decoder_hidden_dim=128):
        super(Actor, self).__init__()
        # Using a simpler attention mechanism for the policy head
        self.key_proj = nn.Linear(hidden_node_dim, decoder_hidden_dim, bias=False)
        self.query_proj = nn.Linear(hidden_node_dim, decoder_hidden_dim, bias=False)

    def forward(self, node_embeddings, pooled_embedding, mask):
        # The query is based on the global state of the graph
        query = self.query_proj(pooled_embedding).unsqueeze(1)  # [B, 1, H]

        # The keys are the individual node representations
        keys = self.key_proj(node_embeddings)  # [B, N, H]

        # Calculate compatibility scores (attention)
        compatibility = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(keys.size(-1))
        compatibility = torch.tanh(compatibility) * 10.0
        compatibility = compatibility.squeeze(1)  # [B, N]

        if mask is not None:
            compatibility = compatibility.masked_fill(mask, float('-inf'))

        probs = F.softmax(compatibility, dim=-1)

        dist = Categorical(probs=probs)
        return dist


class CriticQ(nn.Module):
    """
    The SAC Critic (Q-value) network.
    It estimates the value of a state-action pair: Q(s, a).
    """

    def __init__(self, hidden_node_dim):
        super(CriticQ, self).__init__()
        # Input will be the pooled graph embedding (state) + the chosen node's embedding (action)
        self.fc1 = nn.Linear(hidden_node_dim * 2, hidden_node_dim)
        self.fc2 = nn.Linear(hidden_node_dim, hidden_node_dim // 2)
        self.fc3 = nn.Linear(hidden_node_dim // 2, 1)

    def forward(self, pooled_embedding, action_embedding):
        # Concatenate the state and action representations
        x = torch.cat([pooled_embedding, action_embedding], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class AgentSAC:
    def __init__(self,
                 # Encoder params
                 raw_node_feature_dim, demand_feature_dim, hidden_node_dim,
                 input_edge_dim, hidden_edge_dim, conv_layers,
                 # SAC params
                 n_nodes, learning_rate=5e-5,
                 gamma=0.98, tau=0.005, replay_buffer_capacity=50000):

        self.gamma = gamma
        self.tau = tau
        self.n_nodes = n_nodes

        # Re-use the GNN Encoder from the PPO implementation
        self.encoder = Encoder(raw_node_feature_dim, demand_feature_dim, hidden_node_dim,
                               input_edge_dim, hidden_edge_dim, conv_layers).to(device)

        # Actor Network
        self.actor = Actor(hidden_node_dim=hidden_node_dim, n_heads=4, n_nodes=n_nodes).to(device)

        # Critic Networks
        self.critic1 = CriticQ(hidden_node_dim).to(device)
        self.critic2 = CriticQ(hidden_node_dim).to(device)
        self.critic1_target = CriticQ(hidden_node_dim).to(device)
        self.critic2_target = CriticQ(hidden_node_dim).to(device)

        # Initialize target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=learning_rate)

        # Automatic Entropy Tuning (alpha)
        self.target_entropy = -torch.tensor(1.0 / n_nodes).log().to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-4) 
        self.alpha = self.log_alpha.exp()

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
    def save_models(self, file_path_prefix):
        """Saves the state_dict of all networks."""
        print(f"💾 Saving models to {file_path_prefix}_*.pth")
        torch.save(self.encoder.state_dict(), f'{file_path_prefix}_encoder.pth')
        torch.save(self.actor.state_dict(), f'{file_path_prefix}_actor.pth')
        
        # Assuming your critics are named self.critic and self.target_critic
        # If they have different names, please adjust accordingly.
        if hasattr(self, 'critic'):
            torch.save(self.critic.state_dict(), f'{file_path_prefix}_critic.pth')
        if hasattr(self, 'target_critic'):
            torch.save(self.target_critic.state_dict(), f'{file_path_prefix}_target_critic.pth')
    
    def select_action(self, state_batch):
        with torch.no_grad():
            # Get node embeddings from the shared encoder
            node_embeddings_flat = self.encoder(state_batch)
            pooled_embedding = global_mean_pool(node_embeddings_flat, state_batch.batch)

            batch_size = state_batch.num_graphs
            node_embeddings_batch = node_embeddings_flat.view(batch_size, self.n_nodes, -1)

            # Get action distribution from actor
            dist = self.actor(node_embeddings_batch, pooled_embedding, state_batch.mask.view(batch_size, -1))
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu().numpy(), log_prob

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return None, None, None  # Not enough samples to train

        # 1. Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # --- Get Embeddings for Critic Target ---
        with torch.no_grad():
            next_node_embed_flat = self.encoder(next_states)
            next_pooled_embed = global_mean_pool(next_node_embed_flat, next_states.batch)
            next_node_embed_batch = next_node_embed_flat.view(batch_size, self.n_nodes, -1)

            # 2. Compute Critic Target Q value
            next_dist = self.actor(next_node_embed_batch, next_pooled_embed, next_states.mask.view(batch_size, -1))
            # Use probabilities for sampling next actions for a more stable target
            next_action_probs = next_dist.probs
            next_actions_sampled = torch.multinomial(next_action_probs, 1).squeeze(-1)
            next_log_prob = next_dist.log_prob(next_actions_sampled)
            
            # Get embeddings for the next actions
            next_action_embed = torch.stack([next_node_embed_batch[i, next_actions_sampled[i], :] for i in range(batch_size)])

            target_q1 = self.critic1_target(next_pooled_embed, next_action_embed)
            target_q2 = self.critic2_target(next_pooled_embed, next_action_embed)
            target_q_min = torch.min(target_q1, target_q2)

            target_q = rewards + (1 - dones) * self.gamma * (target_q_min - self.alpha.detach() * next_log_prob.unsqueeze(1))

        # --- 3. Update Critic Networks ---
        # Get current state embeddings (these will be detached for the critic update)
        node_embed_flat = self.encoder(states)
        pooled_embed = global_mean_pool(node_embed_flat, states.batch)
        node_embed_batch = node_embed_flat.view(batch_size, self.n_nodes, -1)
        
        # Get embeddings for the actions taken
        action_embed = torch.stack([node_embed_batch[i, actions[i].item(), :] for i in range(batch_size)])

        # Critic 1 update
        current_q1 = self.critic1(pooled_embed.detach(), action_embed.detach())
        critic1_loss = F.mse_loss(current_q1, target_q)
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optim.step()

        # Critic 2 update
        current_q2 = self.critic2(pooled_embed.detach(), action_embed.detach())
        critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optim.step()

        total_critic_loss = critic1_loss + critic2_loss

        # --- 4. Update Actor Network and Encoder ---
        # Use embeddings with gradients attached for this part
        dist = self.actor(node_embed_batch, pooled_embed, states.mask.view(batch_size, -1))
        
        # We need to re-calculate Q-values with gradients flowing through the actor and encoder
        # We use the raw probabilities from the policy for this, not a new sample
        probs_pi = dist.probs # [B, N]
        # To get Q-values for all actions, we need to repeat the state embedding
        # and combine it with all action embeddings
        pooled_embed_exp = pooled_embed.unsqueeze(1).expand(-1, self.n_nodes, -1) # [B, N, H]
        
        # Get Q-values for all possible actions
        q1_pi_all = self.critic1(pooled_embed_exp.reshape(-1, pooled_embed.shape[-1]), node_embed_batch.reshape(-1, node_embed_batch.shape[-1])).view(batch_size, -1)
        q2_pi_all = self.critic2(pooled_embed_exp.reshape(-1, pooled_embed.shape[-1]), node_embed_batch.reshape(-1, node_embed_batch.shape[-1])).view(batch_size, -1)
        q_pi_min_all = torch.min(q1_pi_all, q2_pi_all)

        # Calculate the actor loss as the expectation over the policy distribution
        actor_loss = (probs_pi * (self.alpha.detach() * torch.log(probs_pi + 1e-8) - q_pi_min_all)).sum(dim=1).mean()
        
        log_probs_pi_for_alpha = (probs_pi * torch.log(probs_pi + 1e-8)).sum(dim=-1)

        # This update affects both the actor and the encoder
        self.encoder_optim.zero_grad()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.encoder_optim.step()
        self.actor_optim.step()

        # --- 5. Update Alpha (Entropy Coefficient) ---
        alpha_loss = -(self.log_alpha * (log_probs_pi_for_alpha.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        # --- 6. Soft update target networks ---
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)

        return actor_loss.item(), total_critic_loss.item(), self.alpha.item()

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)