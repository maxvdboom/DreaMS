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
    """Enhanced callback to manage memory during training."""
    
    def __init__(self, clear_cache_every_n_steps=25, aggressive_cleanup=True):
        self.clear_cache_every_n_steps = clear_cache_every_n_steps
        self.step_count = 0
        self.aggressive_cleanup = aggressive_cleanup
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Clear cache before each batch for maximum memory efficiency
        if self.aggressive_cleanup:
            torch.cuda.empty_cache()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.step_count += 1
        
        # More frequent cache clearing
        if self.step_count % self.clear_cache_every_n_steps == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # Clear variables that might hold references
        if self.aggressive_cleanup:
            del batch
            if outputs is not None:
                del outputs
            torch.cuda.empty_cache()
    
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        # Clear cache before validation batches
        torch.cuda.empty_cache()
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Clear cache after each validation batch
        torch.cuda.empty_cache()
        if self.aggressive_cleanup:
            del batch
            if outputs is not None:
                del outputs
    
    def on_validation_epoch_start(self, trainer, pl_module):
        # Major cleanup before validation
        torch.cuda.empty_cache()
        gc.collect()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Force garbage collection and cache clearing after validation
        torch.cuda.empty_cache()
        gc.collect()
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Clear cache before saving checkpoint
        torch.cuda.empty_cache()
        gc.collect()

def get_memory_optimized_args(original_args):
    """Modify training arguments for memory optimization."""
    # Store original batch size for gradient accumulation
    original_args.effective_batch_size = original_args.batch_size
    
    # Reduce batch size to 1 for maximum memory efficiency
    original_args.batch_size = 1
    
    # Calculate gradient accumulation steps to maintain effective batch size
    original_args.accumulate_grad_batches = original_args.effective_batch_size
    
    # Disable validation output storage to save memory
    original_args.store_val_out_dir = None
    
    # Reduce validation frequency
    original_args.val_check_interval = 0.25  # Every 25% of epoch instead of 10%
    
    # Enable gradient checkpointing if available
    original_args.gradient_clip_val = 1.0
    
    # Reduce number of workers to save CPU memory
    original_args.num_workers_data = min(2, original_args.num_workers_data)
    
    # Disable some memory-intensive features during training
    original_args.no_val = False  # Keep validation but make it more memory efficient
    
    return original_args

# Usage example:
# from memory_optimized_train import MemoryOptimizedCallback, get_memory_optimized_args
# args = get_memory_optimized_args(args)
# callbacks.append(MemoryOptimizedCallback(clear_cache_every_n_steps=25))
