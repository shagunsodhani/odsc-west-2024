# Code for FSDP tutorial at ODSC West 2024

## Setup Instructions

1. Create a new virtual environment:
```bash
conda create -n odsc python=3.11 -y
conda activate odsc
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Training Commands

### Basic DDP Training
```bash
cd src
PYTHONPATH=. torchrun linear_example/train.py --data_parallel ddp --batch_size 8192
```
This runs the most basic distributed training setup using DistributedDataParallel (DDP). Good starting point for understanding distributed training. This would produce logs like 

```
...
{'stage': 'after model creation', 'rank': 0, 'log_key': 'memory', 'cuda_memory_allocated_GB': 9.00048828125, 'cuda_max_memory_allocated_GB': 9.00048828125, 'cuda_memory_reserved_GB': 9.001953125, 'cuda_max_memory_reserved_GB': 9.001953125, 'cpu_memory_rss_GB': 0.57122802734375}
...
{'stage': 'after warmup', 'rank': 0, 'log_key': 'memory', 'cuda_memory_allocated_GB': 25.017333984375, 'cuda_max_memory_allocated_GB': 26.595459938049316, 'cuda_memory_reserved_GB': 29.015625, 'cuda_max_memory_reserved_GB': 29.015625, 'cpu_memory_rss_GB': 0.9235191345214844}
```

### Basic FSDP Training
```bash
PYTHONPATH=. torchrun linear_example/train.py --data_parallel fsdp --batch_size 8192
```
Uses FullyShardedDataParallel (FSDP) with default settings. Typically provides better memory efficiency than DDP.

This would produce logs like 

```
...
{'stage': 'after model creation', 'rank': 0, 'log_key': 'memory', 'cuda_memory_allocated_GB': 9.00048828125, 'cuda_max_memory_allocated_GB': 9.00048828125, 'cuda_memory_reserved_GB': 9.001953125, 'cuda_max_memory_reserved_GB': 9.001953125, 'cpu_memory_rss_GB': 0.5711441040039062}
...
{'stage': 'after warmup', 'rank': 0, 'log_key': 'memory', 'cuda_memory_allocated_GB': 3.0159912109375, 'cuda_max_memory_allocated_GB': 11.016174793243408, 'cuda_memory_reserved_GB': 16.15625, 'cuda_max_memory_reserved_GB': 16.15625, 'cpu_memory_rss_GB': 0.9244613647460938}
```


### Memory-Optimized FSDP Training
```bash
PYTHONPATH=. torchrun linear_example/train.py \
    --data_parallel fsdp \
    --batch_size 8192 \
    --cpu_offload
```
Optimizes for maximum memory efficiency by enabling CPU offloading


### Performance-Optimized FSDP Training
```bash
PYTHONPATH=. torchrun linear_example/train.py \
    --data_parallel fsdp \
    --batch_size 8192 \
    --forward_prefetch \
    --backward_prefetch BACKWARD_POST
```
Optimizes for maximum training speed by:
- Enabling forward prefetching
- Using post-backward prefetching

## Monitoring

All training runs will:
1. Generate TensorBoard profiling data in a directory named based on your configuration
2. Print memory usage statistics at key points during training
3. Profile both CPU and CUDA operations

To view the profiling data:
```bash
tensorboard --logdir .
```

## Configuration Quick Reference

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| data_parallel | ddp, fsdp | fsdp | Distributed training strategy |
| auto_wrap_policy | module_wrap, size_based, none | module_wrap | How to wrap model layers |
| backward_prefetch | BACKWARD_PRE, BACKWARD_POST, NONE | BACKWARD_PRE | When to prefetch for backward pass |
| batch_size | integer | 8192 | Training batch size |
| min_num_params | integer | 100000000 | Minimum params for size-based wrapping |

## Visualizing Memory Usage

The script outputs memory usage at key points:
- After data creation
- After model creation
- After warmup phase

Example output interpretation:
```
{'stage': 'after warmup', 'rank': 0, 'log_key': 'memory', 'cuda_memory_allocated_GB': 25.017333984375, 'cuda_max_memory_allocated_GB': 26.595459938049316, 'cuda_memory_reserved_GB': 29.015625, 'cuda_max_memory_reserved_GB': 29.015625, 'cpu_memory_rss_GB': 0.9235191345214844}
```

This helps track memory usage patterns and identify potential bottlenecks.