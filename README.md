# Quantum Routing with Deep Reinforcement Learning

This project is a simulation framework for comparing routing algorithms in quantum networks. It uses a sophisticated Deep Reinforcement Learning (DRL) agent to find optimal paths for entanglement distribution and benchmarks its performance against classical algorithms like Dijkstra's and Random Walk across various network topologies.

## 📋 Table of Contents

- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Execution](#installation--execution)
- [📈 Viewing Results](#-viewing-results)
- [🔬 Core Technology](#-core-technology)
- [🔧 Running a Single Experiment](#-running-a-single-experiment)
- [📁 Project Structure](#-project-structure)
- [📄 License](#-license)

## 🚀 Getting Started

To get the project running, follow these steps.

### Prerequisites

- Python 3.9+
- Git

### Installation & Execution

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Navigate into the source directory**
   ```bash
   cd src
   ```

4. **Make the experiment script executable** (only needed once)
   ```bash
   chmod +x run_experiments.sh
   ```

5. **Run the full experiment suite**
   ```bash
   ./run_experiments.sh
   ```

## 📈 Viewing Results

After the script finishes, all logs, comparative plots, and summary reports will be available in the `outputs/` directory at the root of the project.

**Note:** The `run_experiments.sh` script automatically calls the analysis script upon completion.

## 🔬 Core Technology

This project integrates several modern technologies to achieve its goals:

- **DRL Agent:**
  - Noisy Dueling DQN: For more stable and efficient Q-value estimation and learned exploration.
  - Prioritized Experience Replay: To focus training on the most informative transitions.

- **Simulation:**
  - Python: Core language for simulation logic.
  - NetworkX: For creating, manipulating, and studying complex network structures.

- **Machine Learning:**
  - PyTorch: Framework used to build and train the DRL agent.

- **Data Analysis:**
  - pandas: For manipulation and aggregation of experiment logs.
  - Matplotlib & Seaborn: For generating high-quality comparative plots and visualizations.

## 🔧 Running a Single Experiment

To run a specific simulation without executing the full suite, use `main.py` from within the `src` directory:

```bash
# Ensure you are in the 'src' directory
cd src

# Example: Run DRL on the 'nsfnet' topology for 2000 episodes
python main.py --topology nsfnet --algorithm drl --episodes 2000
```

## 📁 Project Structure

The project is organized into a clean `src` and `outputs` structure to separate source code from generated files.

```
.
├── src/
│   ├── main.py                 # Main script to run a single experiment
│   ├── run_experiments.sh      # Shell script to automate all experiments
│   └── ...                     # All other source code files
└── outputs/
    ├── logs/                   # Directory for experiment CSV logs
    └── analysis_results/       # Directory for plots and reports
```

## 📄 License

This project is licensed under the terms of the license agreement included in the `LICENSE` file.
