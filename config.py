# config.py
"""
Central configuration file for the Quantum Routing simulation.
(UPDATED FOR HIGHER PERFORMANCE)
"""

config = {
    # ── Core quantum parameters ─────────────────────────────
    "num_nodes": 14,
    "decoherence_rate": 0.001,
    "fidelity_threshold": 0.75,
    "entanglement_lifetime": 150,
    "quantum_memory_size": 100,
    "initial_fidelity": 0.98,

    # ── Enhanced DQN parameters (UPDATED) ──────────────────
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay_steps": 2500,     # Faster decay to encourage exploitation sooner
    "epsilon_min": 0.01,
    "batch_size": 256,
    "learning_rate": 1e-4,           # Lower learning rate for more stable updates
    "hidden_size": 512,
    "target_update_freq": 2500,      # More stable target network
    "grad_clip": 1.0,
    "replay_buffer_size": 20000,
    "min_buffer_fill": 2000,
    "store_frequency": 1,
    "training_frequency": 4,

    # --- PER Parameters (NEW) ---
    "per_alpha": 0.6,                # Priority exponent
    "per_beta_start": 0.4,           # Initial importance sampling weight
    "per_beta_frames": 100000,       # Frames to anneal beta to 1.0

    # ── Routing parameters ──────────────────────────────────
    "max_path_length": 10,
    "min_fidelity": 0.65,
    "success_bonus": 30,
    "path_penalty": 0.5,
    "purification_rounds": 3,
    "ema_alpha": 0.99,

    # ── Adaptive threshold and load balancing ───────────────
    "adaptive_threshold": True,
    "load_balancing": True,
    "edge_validation": True,
    "load_penalty": 0.2,
    "min_edge_count": 2,

    # ── Noise modelling ─────────────────────────────────────
    "noise_std": 0.02,
    "edge_failure_prob": 0.02,
    "memory_error_prob": 0.02,
    "decoherence_variance": 0.0003,
    "min_decoherence": 0.0005,
    "max_decoherence": 0.002,

    # ── RL diagnostics ─────────────────────────────────────
    "diagnostics": {
        "q_value_window": 100,
        "td_error_window": 100,
        "print_frequency": 100
    }
}