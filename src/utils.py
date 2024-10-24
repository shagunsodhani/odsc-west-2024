"""
Training Utilities Module
------------------------
Collection of utility functions for PyTorch distributed training monitoring and setup.

This module provides functions for:
- Memory usage tracking (CPU and GPU)
- Tensor size calculation
- Reproducible seed setting

- PyTorch Memory Management: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
- CUDA Memory Documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
"""

import random

import psutil
import torch
from dist_utils import get_rank

def get_memory_usage(prefix: str = "") -> dict[str, str | int | float]:
    """Get detailed memory usage statistics for both CPU and GPU.
    
    Collects memory statistics including:
    - Currently allocated CUDA memory
    - Peak allocated CUDA memory
    - Currently reserved CUDA memory
    - Peak reserved CUDA memory
    - CPU RSS (Resident Set Size)
    
    Args:
        prefix: Optional string to identify the measurement point
        
    Returns:
        Dictionary containing memory usage statistics
        
    Example:
        >>> # Check memory after model creation
        >>> stats = get_memory_usage("post_model_init")
        >>> print(f"GPU Memory Used: {stats['cuda_memory_allocated_GB']:.2f} GB")
    
    Notes:
        - Memory values are converted to gigabytes
        - CUDA memory tracking requires PyTorch with CUDA support
        - RSS represents the non-swapped physical memory used
    """
    memory_stats = {
        "stage": prefix,
        "rank": get_rank(),
        "log_key": "memory",  # Used for log filtering
        
        # Current GPU memory allocated by tensors
        "cuda_memory_allocated_GB": torch.cuda.memory_allocated(),
        
        # Peak GPU memory allocated by tensors
        "cuda_max_memory_allocated_GB": torch.cuda.max_memory_allocated(),
        
        # Current GPU memory managed by caching allocator
        "cuda_memory_reserved_GB": torch.cuda.memory_reserved(),
        
        # Peak GPU memory managed by caching allocator
        "cuda_max_memory_reserved_GB": torch.cuda.max_memory_reserved(),
        
        # Physical memory used by process
        "cpu_memory_rss_GB": psutil.Process().memory_info().rss,
    }
    
    # Convert byte values to gigabytes
    for key in memory_stats:
        if key.endswith("_GB"):
            memory_stats[key] = memory_stats[key] / (1024**3)
    
    return memory_stats


def set_seed(seed: int, cuda_deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.
    
    Sets random seeds for:
    - PyTorch CPU operations
    - Python's random module
    - CUDA operations (if available)
    
    Args:
        seed: Integer seed for random number generation
        cuda_deterministic: If True, configure CUDA to be deterministic
            (may impact performance)
            
    Example:
        >>> set_seed(42)  # Basic reproducibility
        >>> set_seed(42, cuda_deterministic=True)  # Full determinism
        
    Notes:
        - Setting CUDA operations to be deterministic may significantly
          impact performance
        - Even with seeds set, some PyTorch operations may still be
          non-deterministic on GPU
    """
    torch.manual_seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        if cuda_deterministic:
            # Configure CUDA for full determinism
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
