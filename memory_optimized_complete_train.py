#!/usr/bin/env python3
"""
Complete memory-optimized training script that combines all strategies.
Usage: python memory_optimized_complete_train.py
"""

import os
import sys
import torch
import gc
from pathlib import Path

# Add memory optimizations before importing other modules
torch.backends.cudnn.benchmark = False  # Reduce memory fragmentation
torch.backends.cudnn.deterministic = True

# Import DreaMS modules
sys.path.append(str(Path(__file__).parent))
from dreams.training.train_argparse import parse_args
from memory_optimized_train import MemoryOptimizedCallback, get_memory_optimized_args
from memory_optimized_head import apply_memory_optimizations
from gradient_accumulation import GradientAccumulationStrategy, create_memory_efficient_trainer


def setup_memory_optimizations():
    """Set up all memory optimizations."""
    # Apply monkey patches
    apply_memory_optimizations()
    
    # Set environment variables for memory efficiency
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Set torch settings
    torch.set_float32_matmul_precision('medium')  # Use medium precision for memory savings
    
    print("Applied all memory optimizations")


def main():
    # Setup memory optimizations first
    setup_memory_optimizations()
    
    # Parse arguments
    args = parse_args()
    
    # Apply memory-optimized argument modifications
    args = get_memory_optimized_args(args)
    args, accumulate_steps = GradientAccumulationStrategy.modify_trainer_args(args, accumulate_grad_batches=4)
    
    # Import and run the original training logic with modifications
    import dreams.training.train as original_train
    
    # Override specific functions for memory efficiency
    original_main = original_train.main
    
    def memory_optimized_main(args):
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Call original main with our modifications
        return original_main(args)
    
    # Replace the main function
    original_train.main = memory_optimized_main
    
    # Run training
    memory_optimized_main(args)


if __name__ == '__main__':
    main()
