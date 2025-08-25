#!/usr/bin/env python3
"""
Extreme memory optimization strategies for DreaMS fine-tuning.
Use when standard optimization is not sufficient.
"""

import gc
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import warnings

class ExtremeMemoryOptimizedCallback(Callback):
    """Ultra-aggressive memory management callback."""
    
    def __init__(self, 
                 clear_cache_every_step=True,
                 force_gc_every_n_steps=10,
                 disable_amp_autocast=False):
        self.clear_cache_every_step = clear_cache_every_step
        self.force_gc_every_n_steps = force_gc_every_n_steps
        self.step_count = 0
        self.disable_amp_autocast = disable_amp_autocast
    
    def setup(self, trainer, pl_module, stage):
        """Setup optimizations when training starts."""
        # Disable autocast if requested (saves memory but may affect training)
        if self.disable_amp_autocast and hasattr(torch.cuda.amp, 'autocast'):
            warnings.warn("Disabling AMP autocast for extreme memory savings. May affect training stability.")
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Aggressive pre-batch cleanup
        torch.cuda.empty_cache()
        if batch_idx % self.force_gc_every_n_steps == 0:
            gc.collect()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.step_count += 1
        
        # Clear cache every step if enabled
        if self.clear_cache_every_step:
            torch.cuda.empty_cache()
        
        # Force garbage collection periodically
        if self.step_count % self.force_gc_every_n_steps == 0:
            gc.collect()
            torch.cuda.empty_cache()
        
        # Manually delete references
        del batch
        if outputs is not None:
            del outputs
    
    def on_validation_start(self, trainer, pl_module):
        # Major cleanup before validation
        torch.cuda.empty_cache()
        gc.collect()
    
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        torch.cuda.empty_cache()
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        torch.cuda.empty_cache()
        del batch
        if outputs is not None:
            del outputs
    
    def on_validation_end(self, trainer, pl_module):
        # Major cleanup after validation
        torch.cuda.empty_cache()
        gc.collect()

def get_extreme_memory_args(original_args):
    """Apply extreme memory optimization settings."""
    # Force smallest possible batch size
    original_args.effective_batch_size = getattr(original_args, 'batch_size', 2)
    original_args.batch_size = 1
    original_args.accumulate_grad_batches = original_args.effective_batch_size
    
    # Reduce validation frequency dramatically
    original_args.val_check_interval = 0.5  # Only twice per epoch
    
    # Minimize workers
    original_args.num_workers_data = 1
    
    # Disable memory-intensive features
    original_args.store_val_out_dir = None
    original_args.store_probing_pred = False
    
    # Reduce checkpoint frequency
    original_args.save_top_k = 1
    
    # Enable gradient clipping for stability with small batches
    original_args.gradient_clip_val = 1.0
    
    # Reduce logging frequency
    original_args.log_every_n_steps = max(50, original_args.log_every_n_steps)
    
    return original_args

def enable_memory_efficient_attention():
    """Enable memory-efficient attention if available."""
    try:
        # Try to enable flash attention or other memory-efficient variants
        import torch.nn.functional as F
        if hasattr(F, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
            print("Enabled Flash Attention for memory efficiency")
    except:
        pass

def setup_extreme_memory_environment():
    """Setup environment variables for maximum memory efficiency."""
    import os
    
    # PyTorch memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
    
    # Disable CUDA memory pool if available
    try:
        torch.cuda.set_per_process_memory_fraction(0.95)  # Leave some memory for system
    except:
        pass
    
    # Enable memory efficient attention
    enable_memory_efficient_attention()
    
    print("Applied extreme memory optimization environment settings")

# Example usage:
# from extreme_memory_optimization import ExtremeMemoryOptimizedCallback, get_extreme_memory_args, setup_extreme_memory_environment
# 
# setup_extreme_memory_environment()
# args = get_extreme_memory_args(args)
# callbacks.append(ExtremeMemoryOptimizedCallback(
#     clear_cache_every_step=True,
#     force_gc_every_n_steps=5
# ))
