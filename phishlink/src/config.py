# config.py
# Default configuration for the Phishlink project.

CONFIG = {
    "seed": 42,
    "max_len": 200,
    "embedding_dim": 64,
    "batch_size": 128,  # Fallback to 32 will be handled in runtime if GPU is absent
    "lr": 1e-3,
    "epochs": 30,
    "dropout": 0.2,
    "patience": 5,
    "dataset_ratios": (0.8, 0.1, 0.1) # Train, Val, Test
}
