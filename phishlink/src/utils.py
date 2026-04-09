import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """Gets a logger that writes to console and optionally a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def save_metadata(path: str, metadata: dict):
    """Save metadata to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metadata["timestamp"] = datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)
