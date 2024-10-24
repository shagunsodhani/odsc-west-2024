"""
Utility functions for PyTorch distributed computing operations.

This module provides helper functions for managing distributed training processes
in PyTorch. It includes utilities for checking process rank, world size, and
identifying the main process in a distributed setup.

References:
    - PyTorch Distributed Overview:
      https://pytorch.org/docs/stable/distributed.html
    - PyTorch DistributedDataParallel:
      https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel
    - PyTorch Multi-GPU Training:
      https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

Example:
    >>> # Initialize distributed process group
    >>> torch.distributed.init_process_group(backend='nccl')
    >>> rank = get_rank()
    >>> world_size = get_world_size()
    >>> if is_main_process():
    >>>     print(f"Main process in world of size {world_size}")
"""

import torch.distributed as dist

def get_rank() -> int:
    """
    Get the rank of current process in distributed computing setup.
    
    The rank is a unique identifier for each process in the distributed
    setup, with 0 being the main process.
    
    Returns:
        int: The rank of current process (0 if not distributed)
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size() -> int:
    """
    Get total number of processes in distributed computing setup.
    
    World size represents the total number of processes participating
    in the distributed training.
    
    Returns:
        int: Total number of processes (1 if not distributed)
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

def is_main_process() -> bool:
    """
    Check if current process is the main process (rank 0).
    
    The main process is typically used for logging, saving checkpoints,
    and other operations that should only be performed once across
    all distributed processes.
    
    Returns:
        bool: True if current process is main process, False otherwise
    """
    return get_rank() == 0