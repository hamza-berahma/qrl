import random
import heapq
import networkx as nx

def drl_select_action(node, state, valid_actions):
    """Uses the node's DRL agent to select an action."""
    return node.select_action(state, valid_actions)

def random_walk_select_action(node, state, valid_actions):
    """Selects a random valid action."""
    return random.choice(valid_actions)

def quantum_dijkstra_select_action(node, state, valid_actions):
    """
    Selects the next hop based on the path with the highest fidelity,
    acting as a quantum-aware version of Dijkstra's algorithm.
    This is a simplified, greedy, next-hop version.
    """
    best_neighbor = None
    max_fidelity = -1

    # Find the neighbor with the highest direct link fidelity
    for neighbor in valid_actions:
        try:
            fidelity = node.graph.edges[node.node_id, neighbor]['fidelity']
            if fidelity > max_fidelity:
                max_fidelity = fidelity
                best_neighbor = neighbor
        except (KeyError, nx.NetworkXError):
            continue
            
    # Fallback to random if no valid neighbor is found
    return best_neighbor if best_neighbor is not None else random.choice(valid_actions)

# Dictionary to easily access algorithms by name
ROUTING_ALGORITHMS = {
    "drl": drl_select_action,
    "dijkstra": quantum_dijkstra_select_action,
    "random": random_walk_select_action
}
