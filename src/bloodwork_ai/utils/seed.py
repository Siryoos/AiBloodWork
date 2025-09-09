"""Random seed utilities for reproducibility."""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_random_state(seed: Optional[int] = None) -> np.random.RandomState:
    """
    Get a numpy RandomState object with the given seed.
    
    Args:
        seed: Random seed value. If None, uses current numpy random state.
        
    Returns:
        RandomState object
    """
    if seed is not None:
        return np.random.RandomState(seed)
    return np.random.RandomState()


def get_torch_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Get a PyTorch generator with the given seed.
    
    Args:
        seed: Random seed value. If None, uses current torch random state.
        
    Returns:
        PyTorch generator
    """
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator
