"""
Distributed Training Tutorial
----------------------------
This script demonstrates distributed training in PyTorch using DDP (DistributedDataParallel) 
and FSDP (FullyShardedDataParallel) with various optimization strategies.

References:
- FSDP Documentation: https://pytorch.org/docs/stable/fsdp.html
- DDP Documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel

"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from dist_utils import is_main_process
from model import MLP
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity, record_function
from utils import get_memory_usage, set_seed


@dataclass
class Config:
    """Configuration class for training parameters.
    
    Attributes:
        batch_size (int): Size of training batch
        input_size (int): Input dimension size
        hidden_size (int): Hidden layer dimension size
        output_size (int): Output dimension size
        data_parallel (str): Type of data parallelism ('ddp' or 'fsdp')
        seed (int): Random seed for reproducibility
        auto_wrap_policy (Optional[str]): FSDP wrapping strategy
        forward_prefetch (bool): Enable forward prefetching in FSDP
        backward_prefetch (BackwardPrefetch): Backward prefetch strategy
        cpu_offload (bool): Enable CPU offloading in FSDP
        min_num_params (int): Minimum parameters for size-based wrapping
    """
    
    batch_size: int
    input_size: int
    hidden_size: int
    output_size: int
    data_parallel: str
    seed: int
    auto_wrap_policy: Optional[str]
    forward_prefetch: bool
    backward_prefetch: BackwardPrefetch
    cpu_offload: bool
    min_num_params: int

    @classmethod
    def build(
        cls,
        batch_size: int,
        input_size: int,
        data_parallel: str,
        auto_wrap_policy: str = "module_wrap",
        forward_prefetch: bool = False,
        backward_prefetch: str = "BACKWARD_PRE",
        cpu_offload: bool = True,
        min_num_params: int = int(1e8),
    ) -> "Config":
        """Factory method to create a Config instance with default values.
        
        Args:
            batch_size: Training batch size
            data_parallel: Parallelization strategy ('ddp' or 'fsdp')
            auto_wrap_policy: FSDP wrapping policy
            forward_prefetch: Enable forward prefetching
            backward_prefetch: Backward prefetch strategy
            cpu_offload: Enable CPU offloading
            min_num_params: Minimum parameters threshold
        
        Returns:
            Config: Configured instance
        """
        return cls(
            batch_size=batch_size,
            input_size=input_size,  # Fixed architecture sizes
            hidden_size=input_size,
            output_size=input_size,
            data_parallel=data_parallel,
            seed=42,
            auto_wrap_policy=auto_wrap_policy if auto_wrap_policy != "none" else None,
            forward_prefetch=forward_prefetch,
            backward_prefetch=BackwardPrefetch[backward_prefetch],
            cpu_offload=cpu_offload,
            min_num_params=min_num_params,
        )


def train_step(
    data: torch.Tensor,
    target: torch.Tensor,
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> None:
    """Performs a single training step.
    
    Args:
        data: Input training data
        target: Target values
        model: Neural network model
        optimizer: Optimization algorithm
        criterion: Loss function
    """
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()


def get_wrapped_model(
    model: nn.Module,
    rank: int,
    config: Config,
) -> nn.Module:
    """Wraps the model with the appropriate distributed training strategy.
    
    Args:
        model: Base model to wrap
        rank: Current process rank
        config: Training configuration
    
    Returns:
        nn.Module: Wrapped model ready for distributed training
    """
    if config.data_parallel == "ddp":
        return DDP(model, device_ids=[rank])
    
    # FSDP Configuration
    wrap_policy = None
    if config.auto_wrap_policy == "module_wrap":
        wrap_policy = ModuleWrapPolicy({nn.Linear})
    elif config.auto_wrap_policy == "size_based":
        wrap_policy = size_based_auto_wrap_policy(min_num_params=config.min_num_params)
    
    return FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        forward_prefetch=config.forward_prefetch,
        backward_prefetch=config.backward_prefetch,
        cpu_offload=CPUOffload(offload_params=config.cpu_offload),
    )


def train(rank: int, world_size: int, config: Config) -> None:
    """Main training function for each distributed process.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        config: Training configuration
    """
    set_seed(seed=config.seed)

    # Initialize process group for distributed training
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Generate synthetic data
    data_rng = torch.Generator(device=device)
    data_rng.manual_seed(config.seed + rank)
    data = torch.randn(
        config.batch_size, config.input_size, generator=data_rng, device=device
    )
    target = torch.randn(
        config.batch_size, config.output_size, generator=data_rng, device=device
    )
    
    if is_main_process():
        print(get_memory_usage(prefix="after data creation"))

    # Create and wrap model
    model = MLP(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        num_layers=world_size,
    ).to(device)

    if is_main_process():
        print(f"Base model architecture: {model}")
        print(get_memory_usage(prefix="after model creation"))

    model = get_wrapped_model(model, rank, config)
    
    if is_main_process():
        print(f"Wrapped model: {model}")

    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()

    # Configure profiler output directory
    profiler_output_dir = Path(
        f"{config.data_parallel}_"
        f"awp-{config.auto_wrap_policy}_"
        f"fp-{config.forward_prefetch}_"
        f"bp-{config.backward_prefetch.name}_"
        f"cpu-{config.cpu_offload}_"
        f"minp-{config.min_num_params}"
    )
    profiler_output_dir.mkdir(exist_ok=True)

    if is_main_process():
        print(f"Rank {rank}: Starting training...")

    # Warmup phase
    for _ in range(5):
        train_step(data, target, model, optimizer, criterion)

    if is_main_process():
        print(get_memory_usage(prefix="after warmup"))

    # Training with profiling
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_output_dir),
        record_shapes=True,
    ) as prof:
        for step in range(5):
            with record_function(f"step_{step}"):
                train_step(data, target, model, optimizer, criterion)
            prof.step()

    if is_main_process():
        print(f"Rank {rank}: Finished training...")

    destroy_process_group()


def main():
    """Entry point for the training script."""
    parser = argparse.ArgumentParser(description="Distributed Training Tutorial")
    parser.add_argument(
        "--data_parallel",
        type=str,
        default="fsdp",
        choices=["fsdp", "ddp"],
        help="Distributed training strategy",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8192,
        help="Training batch size",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=16384,
        help="Training batch size",
    )
    parser.add_argument(
        "--auto_wrap_policy",
        type=str,
        default="module_wrap",
        choices=["module_wrap", "size_based", "none"],
        help="FSDP auto wrapping policy",
    )
    parser.add_argument(
        "--forward_prefetch",
        action="store_true",
        help="Enable FSDP forward prefetching",
    )
    parser.add_argument(
        "--backward_prefetch",
        type=str,
        default="BACKWARD_PRE",
        choices=["BACKWARD_PRE", "BACKWARD_POST", "NONE"],
        help="FSDP backward prefetch strategy",
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable FSDP CPU offloading",
    )
    parser.add_argument(
        "--min_num_params",
        type=int,
        default=int(1e8),
        help="Minimum parameters for size-based wrapping",
    )
    
    args = parser.parse_args()
    config = Config.build(
        data_parallel=args.data_parallel,
        batch_size=args.batch_size,
        input_size=args.input_size,
        auto_wrap_policy=args.auto_wrap_policy,
        forward_prefetch=args.forward_prefetch,
        backward_prefetch=args.backward_prefetch,
        cpu_offload=args.cpu_offload,
        min_num_params=args.min_num_params,
    )

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train,
        args=(world_size, config),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()