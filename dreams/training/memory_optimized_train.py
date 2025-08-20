#!/usr/bin/env python3
"""
Memory-optimized training script for DreaMS fine-tuning.
This script provides minimal modifications to reduce memory usage.
"""

import gc
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class MemoryOptimizedCallback(Callback):
    """Callback to manage memory during training."""
    
    def __init__(self, clear_cache_every_n_steps=50):
        self.clear_cache_every_n_steps = clear_cache_every_n_steps
        self.step_count = 0
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.step_count += 1
        if self.step_count % self.clear_cache_every_n_steps == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Clear cache after each validation batch
        torch.cuda.empty_cache()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Force garbage collection and cache clearing after validation
        torch.cuda.empty_cache()
        gc.collect()

def get_memory_optimized_args(original_args):
    """Modify training arguments for memory optimization."""
    # Reduce batch size
    original_args.batch_size = max(1, original_args.batch_size // 2)
    
    # Disable validation output storage to save memory
    original_args.store_val_out_dir = None
    
    # Reduce validation frequency
    original_args.val_check_interval = 0.25  # Every 25% of epoch instead of 10%
    
    # Enable gradient checkpointing if available
    original_args.gradient_clip_val = 1.0
    
    # Reduce number of workers to save CPU memory
    original_args.num_workers_data = min(4, original_args.num_workers_data)
    
    return original_args

# Usage example:
# from memory_optimized_train import MemoryOptimizedCallback, get_memory_optimized_args
# args = get_memory_optimized_args(args)
# callbacks.append(MemoryOptimizedCallback(clear_cache_every_n_steps=25))
