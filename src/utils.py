# utils.py
"""
Utility functions for statistics, memory tracking, logging, and reporting.
"""
import numpy as np
import psutil
import os
import csv
import time
from rich.console import Console
from rich.table import Table

def safe_mean(arr, default=0.0):
    """Safely compute mean, handling empty arrays."""
    return float(np.mean(arr)) if len(arr) > 0 else default

def safe_std(arr, default=0.0):
    """Safely compute standard deviation, handling empty arrays."""
    return float(np.std(arr)) if len(arr) > 0 else default

def safe_agg(arr, func, default=0.0):
    """Generic safe aggregation for numpy operations."""
    try:
        if isinstance(arr, (list, tuple)) and len(arr) == 0:
            return default
        if isinstance(arr, np.ndarray) and arr.size == 0:
            return default
        result = func(arr)
        return float(result) if np.isscalar(result) else result
    except (ValueError, RuntimeWarning):
        return default

def get_memory_usage():
    """Get current memory usage of the program in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def log_episode_data(episode, episode_history, success, out_file="out.csv"):
    """Append current-episode metrics to a CSV log file."""
    row = {
        "episode": episode,
        "reward": episode_history["rewards"][-1],
        "mean_q_value": episode_history["q_values"][-1],
        "td_error": episode_history["td_errors"][-1],
        "path_length": episode_history["path_lengths"][-1],
        "fidelity": episode_history["fidelities"][-1],
        "success": int(success)
    }
    write_header = not os.path.exists(out_file)
    with open(out_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def generate_final_report(network, history, start_time, start_memory):
    """Generates and saves a final summary report of the training session."""
    console = Console()
    end_time = time.time()
    total_time = end_time - start_time
    end_memory = get_memory_usage()

    report_path = "final_training_report.txt"

    with open(report_path, "w") as f:
        f.write("====== Final Training Report ======\n\n")
        f.write(f"Training Duration: {total_time:.2f} seconds\n")
        f.write(f"Initial Memory: {start_memory:.2f} MB\n")
        f.write(f"Final Memory: {end_memory:.2f} MB\n")
        f.write(f"Memory Change: {end_memory - start_memory:+.2f} MB\n\n")

        # Performance Metrics
        success_rate = safe_mean([1 if s > 0.5 else 0 for s in history['fidelities']])
        avg_reward = safe_mean(history['rewards'][-100:])
        avg_fidelity = safe_mean([f for f in history['fidelities'] if f > 0][-100:])
        avg_path_length = safe_mean([p for p in history['path_lengths'] if p > 1][-100:])

        f.write("------ Performance Summary (Last 100 Episodes) ------\n")
        f.write(f"Overall Success Rate: {success_rate * 100:.2f}%\n")
        f.write(f"Average Reward: {avg_reward:.3f}\n")
        f.write(f"Average Successful Fidelity: {avg_fidelity:.3f}\n")
        f.write(f"Average Path Length: {avg_path_length:.2f}\n\n")

        # RL Metrics
        avg_q_value = safe_mean(history['q_values'][-100:])
        avg_td_error = safe_mean(history['td_errors'][-100:])
        final_epsilon = network.nodes[0].epsilon

        f.write("------ RL Agent Final State ------\n")
        f.write(f"Final Epsilon: {final_epsilon:.4f}\n")
        f.write(f"Average Q-Value: {avg_q_value:.3f}\n")
        f.write(f"Average TD-Error: {avg_td_error:.3f}\n\n")

        # Network State
        top_nodes = sorted(network.stats.node_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        f.write("------ Network State ------\n")
        f.write("Most Used Nodes:\n")
        for node, count in top_nodes:
            f.write(f"  - Node {node}: {count} times\n")

    console.print(f"\n[bold green]Final report saved to {report_path}[/bold green]")
    return report_path