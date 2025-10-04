import numpy as np
import networkx as nx
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, defaultdict

from config import config
from rl_components import DuelingQuantumDQN, PrioritizedReplayBuffer
from topologies import TOPOLOGIES # Import the topologies

# ... (EnhancedQuantumMetrics, NetworkStats, QuantumNode classes remain the same) ...
class EnhancedQuantumMetrics:
    @staticmethod
    def apply_purification(fidelity, rounds=2):
        for _ in range(rounds):
            fidelity = (fidelity**2) / (fidelity**2 + (1 - fidelity)**2)
        return fidelity

    @staticmethod
    def compute_entanglement_fidelity(path, graph):
        if len(path) < 2: return 0.0
        total_fidelity = 1.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if not graph.has_edge(u, v): return 0.0 # Path is broken
            edge = graph.edges[u, v]
            age_factor = edge['age'] / 2000.0
            decoherence = np.exp(-config['decoherence_rate'] * age_factor)
            base_fidelity = edge['fidelity'] * decoherence
            if i > 0:
                purified_fidelity = EnhancedQuantumMetrics.apply_purification(
                    base_fidelity, rounds=config['purification_rounds']
                )
                total_fidelity *= purified_fidelity
            else:
                total_fidelity *= base_fidelity
        return total_fidelity

    @staticmethod
    def compute_reward(path, fidelity, done, success, destination, ema_fidelity=0.0):
        if not done:
            shaping = -0.05
            if len(path) >= 2: return shaping + 0.30 * fidelity**1.5
            return shaping
        if success:
            path_eff = max(0.0, 1.5 - len(path) / config["max_path_length"])
            fidelity_gain = max(0.0, fidelity - ema_fidelity)
            return (1.20 * config["success_bonus"] + 20.0 * fidelity**2 +
                    10.0 * fidelity_gain + 5.0 * path_eff)
        if not path or path[-1] != destination: return -2.0
        return -1.0 * (1.0 - fidelity)

class NetworkStats:
    def __init__(self):
        self.path_lengths, self.path_fidelities, self.rewards = [], [], []
        self.node_usage, self.failure_reasons = defaultdict(int), defaultdict(int)

    def log_episode(self, path, fidelity, reward, success, failure_reason=None):
        if path:
            self.path_lengths.append(len(path))
            self.path_fidelities.append(fidelity)
            for node in path: self.node_usage[node] += 1
        self.rewards.append(reward)
        if not success and failure_reason: self.failure_reasons[failure_reason] += 1

class QuantumNode:
    def __init__(self, node_id, num_nodes, graph=None):
        self.node_id, self.num_nodes, self.graph = node_id, num_nodes, graph
        self.state_dim = num_nodes * 3 + 3
        self.action_dim = num_nodes
        self.dqn = DuelingQuantumDQN(self.state_dim, self.action_dim)
        self.target_dqn = DuelingQuantumDQN(self.state_dim, self.action_dim)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=config["learning_rate"])
        self.memory = PrioritizedReplayBuffer(config["replay_buffer_size"], self.state_dim, alpha=config["per_alpha"])
        self.steps, self.total_episodes = 0, 0
        self._current_epsilon = config["epsilon_start"]
        self.ema_fidelity = 0.0
        self.min_buffer_size = config["min_buffer_fill"]
        self.beta = config["per_beta_start"]

    @property
    def epsilon(self): return self._current_epsilon
    def set_graph(self, graph): self.graph = graph

    def update_epsilon(self):
        self.total_episodes += 1
        decay_rate = -np.log(config["epsilon_end"] / config["epsilon_start"]) / config["epsilon_decay_steps"]
        self._current_epsilon = config["epsilon_end"] + (config["epsilon_start"] - config["epsilon_end"]) * np.exp(-decay_rate * self.total_episodes)
        self._current_epsilon = max(self._current_epsilon, config["epsilon_min"])
        beta_start = config["per_beta_start"]
        self.beta = min(1.0, beta_start + self.total_episodes * (1.0 - beta_start) / config["per_beta_frames"])

    def select_action(self, state, valid_actions):
        if not valid_actions: return None
        if random.random() > self._current_epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.dqn(state_tensor).squeeze()
                valid_indices = [a - 1 for a in valid_actions]
                masked_q = torch.full_like(q_values, -float('inf'))
                masked_q[valid_indices] = q_values[valid_indices]
                return masked_q.argmax().item() + 1
        return random.choice(valid_actions)

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.push(state, action - 1, reward, next_state, done)

    def train_step(self):
        if len(self.memory) < self.min_buffer_size: return None, None
        sample = self.memory.sample(config["batch_size"], self.beta)
        states, actions, rewards, next_states, dones, indices, weights = sample
        q_values = self.dqn(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_actions = self.dqn(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_dqn(next_states).gather(1, next_actions).squeeze(-1)
            target_q = rewards / 10.0 + (1 - dones) * config["gamma"] * next_q
        td_errors = (target_q - q_values).abs()
        loss = (weights * F.smooth_l1_loss(q_values, target_q, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), config["grad_clip"])
        self.optimizer.step()
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        self.steps += 1
        if self.steps % config["target_update_freq"] == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
        return loss.item(), td_errors.mean().item()

class QuantumNetwork:
    """Manages the overall network simulation with a selectable topology."""
    def __init__(self, topology_name="nsfnet"):
        self.graph = TOPOLOGIES[topology_name]()
        self.num_nodes = self.graph.number_of_nodes()
        config["num_nodes"] = self.num_nodes # Update config for dynamic state size
        
        self.nodes = [QuantumNode(i, self.num_nodes, self.graph) for i in range(1, self.num_nodes + 1)]
        self.initialize_edges()
        for node in self.nodes:
            node.set_graph(self.graph)
        self.stats = NetworkStats()
        print(f"Network initialized with '{topology_name}' topology: {self.num_nodes} nodes, {self.graph.number_of_edges()} edges.")

    def initialize_edges(self):
        """Initialize all edges with quantum parameters."""
        for u, v in self.graph.edges():
            self.graph.edges[u, v].update({
                'fidelity': np.random.uniform(0.85, 0.95),
                'success_prob': np.random.uniform(0.7, 0.9),
                'age': 0, 'load': 0, 'capacity': 10, 'usage': 0
            })
    
    # ... (update_network_state, get_valid_neighbors, get_state, etc. remain the same) ...
    def update_network_state(self):
        for u, v in self.graph.edges():
            edge = self.graph.edges[u, v]
            noise = np.random.normal(0, config["noise_std"])
            edge['fidelity'] = np.clip(edge['fidelity'] + noise, 0, 1)
            if np.random.random() < config["edge_failure_prob"]: edge['fidelity'] *= np.random.uniform(0.5, 0.8)
            edge['age'] += 1
            decoherence = config["decoherence_rate"] * (1 + edge['age'] / 2000.0)
            edge['fidelity'] *= (1 - decoherence)
            if edge['fidelity'] < config["min_fidelity"]:
                edge['fidelity'] = np.random.uniform(0.8, 0.95)
                edge['age'] = 0

    def get_valid_neighbors(self, current): return list(self.graph.neighbors(current))

    def get_state(self, current, destination):
        state_size = self.num_nodes * 3 + 3
        state = np.zeros(state_size)
        
        # Slices for features
        fidelities_slice = slice(0, self.num_nodes)
        memory_states_slice = slice(self.num_nodes, self.num_nodes * 2)
        edge_states_slice = slice(self.num_nodes * 2, self.num_nodes * 3)

        for neighbor in self.graph.neighbors(current):
            idx = neighbor - 1
            if 0 <= idx < self.num_nodes:
                edge = self.graph.edges[current, neighbor]
                state[fidelities_slice.start + idx] = edge['fidelity']
                state[edge_states_slice.start + idx] = 1.0
        
        try:
            distance = nx.shortest_path_length(self.graph, current, destination)
            state[-3] = distance / self.num_nodes
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            state[-3] = 1.0
        
        has_direct = float(self.graph.has_edge(current, destination))
        state[-2] = has_direct
        state[-1] = self.graph.edges[current, destination]['fidelity'] if has_direct else 0.0
        return state

    def get_adaptive_threshold(self):
        if not config["adaptive_threshold"]: return config["fidelity_threshold"]
        fidelities = [d['fidelity'] for _, _, d in self.graph.edges(data=True)]
        if not fidelities: return config["fidelity_threshold"]
        avg_fidelity = np.mean(fidelities)
        try:
            connectivity = nx.edge_connectivity(self.graph)
        except nx.NetworkXError: # Handle disconnected graphs
            connectivity = 1
        health_factor = (avg_fidelity * connectivity) / (config["initial_fidelity"] * config.get("min_edge_count", 2))
        return np.clip(config["fidelity_threshold"] * health_factor, 0.5, 0.95)
