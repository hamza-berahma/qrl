import time
import numpy as np
import random
import argparse
import os
import pandas as pd
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn

from config import config
from environment import QuantumNetwork, EnhancedQuantumMetrics
from utils import get_memory_usage, log_episode_data, generate_final_report, safe_mean
from routing_algorithms import ROUTING_ALGORITHMS
from topologies import TOPOLOGIES

def run_training(topology, algorithm, episodes, run_name):
    """Main function to run the DRL training loop with selectable components."""
    console = Console()
    network = QuantumNetwork(topology_name=topology)
    select_action_func = ROUTING_ALGORITHMS[algorithm]
    
    # DRL-specific setup
    if algorithm == "drl":
        console.print("\n[bold yellow]Pre-filling replay buffers for DRL agent...[/bold yellow]")
        for node in network.nodes:
            while len(node.memory) < node.min_buffer_size:
                # Optimized pre-filling
                for _ in range(50):
                    if network.num_nodes < 2: continue
                    source, dest = random.sample(list(network.graph.nodes()), 2)
                    state = network.get_state(source, dest)
                    valid_actions = network.get_valid_neighbors(source)
                    if valid_actions:
                        action = random.choice(valid_actions)
                        next_state = network.get_state(action, dest)
                        node.store_experience(state, action, -0.1, next_state, False)

    start_time = time.time()
    start_memory = get_memory_usage()
    console.print(f"\n[bold green]Starting training: {run_name}[/bold green]")
    history = {'rewards': [], 'q_values': [], 'td_errors': [], 'path_lengths': [], 'fidelities': [], 'success': []}

    with Progress(console=console, expand=True) as progress:
        main_task = progress.add_task(f"[cyan]Training {run_name}", total=episodes)

        for episode in range(episodes):
            if algorithm == "drl":
                for node in network.nodes: node.update_epsilon()
            
            if network.num_nodes < 2: continue
            source, destination = random.sample(list(network.graph.nodes()), 2)
            current = source
            path = [source]
            done = False
            episode_reward = 0
            episode_td_errors = []
            
            while not done and len(path) < config['max_path_length']:
                if not (1 <= current <= len(network.nodes)): break
                node = network.nodes[current - 1]
                
                state = network.get_state(current, destination)
                valid_actions = [n for n in network.get_valid_neighbors(current) if n not in path]
                if not valid_actions: break
                
                action = select_action_func(node, state, valid_actions)
                if action is None: break
                
                next_state = network.get_state(action, destination)
                
                # Store the state from which the action was taken
                state_for_log = state
                
                path.append(action)
                current = action
                done = (current == destination)
                
                fidelity = EnhancedQuantumMetrics.compute_entanglement_fidelity(path, network.graph)
                success = done and (fidelity >= network.get_adaptive_threshold())
                reward = EnhancedQuantumMetrics.compute_reward(path, fidelity, done, success, destination, node.ema_fidelity if algorithm == 'drl' else 0)
                
                episode_reward += reward
                
                if algorithm == "drl":
                    prev_node = network.nodes[path[-2] - 1]
                    prev_node.store_experience(state_for_log, action, reward, next_state, done)
                    loss, td_error = prev_node.train_step()
                    if td_error is not None:
                        episode_td_errors.append(td_error)

            # Log episode results
            final_fidelity = EnhancedQuantumMetrics.compute_entanglement_fidelity(path, network.graph)
            final_success = done and (final_fidelity >= network.get_adaptive_threshold())
            history['rewards'].append(episode_reward)
            history['path_lengths'].append(len(path))
            history['fidelities'].append(final_fidelity)
            history['success'].append(int(final_success))
            history['td_errors'].append(safe_mean(episode_td_errors))
            history['q_values'].append(0) # Placeholder for Q-values
            
            network.stats.log_episode(path, final_fidelity, episode_reward, final_success)
            network.update_network_state()
            progress.update(main_task, advance=1)

    # Save results and generate report
    log_dir = "logs"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{run_name}.csv")
    pd.DataFrame(history).to_csv(log_file, index_label="episode")
    
    console.print(f"\n[bold]Finished training for {run_name}. Log saved to {log_file}[/bold]")
    generate_final_report(network, history, start_time, start_memory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Quantum Routing Simulations.")
    parser.add_argument("--topology", type=str, default="nsfnet", choices=TOPOLOGIES.keys(), help="Network topology to use.")
    parser.add_argument("--algorithm", type=str, default="drl", choices=ROUTING_ALGORITHMS.keys(), help="Routing algorithm to use.")
    parser.add_argument("--episodes", type=int, default=1500, help="Number of training episodes.")
    args = parser.parse_args()

    run_name = f"{args.topology}_{args.algorithm}"
    run_training(args.topology, args.algorithm, args.episodes, run_name)

