#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting Quantum Routing Experiments..."

# Define experiments
TOPOLOGIES=("nsfnet" "grid" "small_world")
ALGORITHMS=("drl" "dijkstra" "random")
EPISODES=2000 # Increased for more thorough training

# Create a directory for logs if it doesn't exist
LOG_DIR="logs"
ANALYSIS_DIR="analysis_results"
mkdir -p $LOG_DIR
mkdir -p $ANALYSIS_DIR

# Run all experiment combinations
for topo in "${TOPOLOGIES[@]}"; do
  for algo in "${ALGORITHMS[@]}"; do
    echo "-----------------------------------------------------"
    echo "RUNNING: Topology: $topo, Algorithm: $algo"
    echo "-----------------------------------------------------"
    python main.py --topology "$topo" --algorithm "$algo" --episodes $EPISODES
  done
done

echo "-----------------------------------------------------"
echo "All experiments complete. Starting analysis..."
echo "-----------------------------------------------------"

# Run the analysis script on all generated log files
python analysis.py $LOG_DIR/*.csv --output_dir $ANALYSIS_DIR

echo "-----------------------------------------------------"
echo "Analysis complete. Plots are saved in the '$ANALYSIS_DIR' directory."
echo "-----------------------------------------------------"
