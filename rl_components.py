# rl_components.py
"""
Core components for the Deep Q-Learning agent.
(UPDATED with Prioritized Replay Buffer, Noisy Dueling DQN)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math

from config import config
from utils import safe_mean, safe_std, safe_agg

# --- Prioritized Replay Buffer (unchanged interface) ---
class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for more efficient learning."""
    def __init__(self, capacity, state_dim, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.priorities = np.zeros(capacity, dtype=np.float32)
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        """Store a transition with max priority."""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.priorities[self.pos] = self.max_priority
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        """Sample a batch, prioritizing experiences with high error."""
        if self.size == 0:
            return None
        
        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        if probs.sum() == 0:
            probs = np.ones_like(prios) / len(prios)
        else:
            probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Importance-sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)

        return (
            torch.FloatTensor(self.states[indices]),
            torch.LongTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.dones[indices]),
            indices,
            weights
        )

    def update_priorities(self, batch_indices, batch_priorities):
        """Update priorities of sampled experiences."""
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + 1e-5
        self.max_priority = max(self.max_priority, np.max(batch_priorities))

    def __len__(self):
        return self.size

# --- Noisy Linear Layer ---
class NoisyLinear(nn.Module):
    """Noisy linear module for learned exploration."""
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon
            )
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

# --- Dueling DQN with Noisy Layers ---
class DuelingQuantumDQN(nn.Module):
    """Dueling DQN using NoisyLinear layers for exploration."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.feature_layer = nn.Sequential(
            NoisyLinear(state_dim, config["hidden_size"]),
            nn.LayerNorm(config["hidden_size"]),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            NoisyLinear(config["hidden_size"], config["hidden_size"]),
            nn.LayerNorm(config["hidden_size"]),
            nn.ReLU(),
            NoisyLinear(config["hidden_size"], 1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(config["hidden_size"], config["hidden_size"]),
            nn.LayerNorm(config["hidden_size"]),
            nn.ReLU(),
            NoisyLinear(config["hidden_size"], action_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# --- RL Diagnostics (unchanged interface) ---
class RLDiagnostics:
    """Enhanced RL-specific learning metrics."""
    def __init__(self):
        self.q_values = []
        self.td_errors = []
        self.losses = []
        self.epsilon_history = []
        self.action_counts = defaultdict(int)
        self.greedy_actions = 0
        self.total_actions = 0

    def log_action(self, action, q_values, epsilon, is_greedy):
        if q_values is None: return
        self.q_values.append({
            'mean': safe_mean(q_values),
            'max': safe_agg(q_values, np.max),
            'std': safe_std(q_values)
        })
        self.action_counts[action] += 1
        self.epsilon_history.append(epsilon)
        if is_greedy: self.greedy_actions += 1
        self.total_actions += 1

    def log_training(self, loss, td_error):
        self.losses.append(float(loss))
        self.td_errors.append(float(td_error))

    def get_summary(self, window=100):
        recent_q = self.q_values[-window:]
        q_stats = {
            'mean': safe_mean([q['mean'] for q in recent_q]),
            'max': safe_mean([q['max'] for q in recent_q]),
            'std': safe_mean([q['std'] for q in recent_q])
        }
        return {
            'q_values': q_stats,
            'td_error': {'mean': safe_mean(self.td_errors[-window:])},
            'loss': {'mean': safe_mean(self.losses[-window:])},
            'epsilon': self.epsilon_history[-1] if self.epsilon_history else 1.0,
            'greedy_ratio': self.greedy_actions / max(1, self.total_actions)
        }
