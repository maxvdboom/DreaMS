#!/usr/bin/env python3
"""
Gradient accumulation strategy to simulate larger batch sizes with less memory.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy


class GradientAccumulationStrategy:
    """
    Implement gradient accumulation to simulate larger batch sizes.
    """
    
    @staticmethod
    def modify_trainer_args(args, accumulate_grad_batches=4):
        """
        Modify trainer arguments for gradient accumulation.
        
        Args:
            args: Training arguments
            accumulate_grad_batches: Number of batches to accumulate gradients over
        """
        # Reduce actual batch size but accumulate gradients
        original_batch_size = args.batch_size
        args.batch_size = max(1, original_batch_size // accumulate_grad_batches)
        
        print(f"Original batch size: {original_batch_size}")
        print(f"New batch size: {args.batch_size}")
        print(f"Gradient accumulation steps: {accumulate_grad_batches}")
        print(f"Effective batch size: {args.batch_size * accumulate_grad_batches}")
        
        return args, accumulate_grad_batches


def create_memory_efficient_trainer(args, callbacks, wandb_logger, accumulate_grad_batches=4):
    """
    Create a memory-efficient trainer with gradient accumulation.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Custom strategy for memory efficiency
    strategy = DDPStrategy(
        find_unused_parameters=True,
        gradient_as_bucket_view=True,  # More memory efficient
        static_graph=False
    ) if args.num_devices > 1 else None
    
    trainer = pl.Trainer(
        strategy=strategy,
        max_epochs=args.max_epochs,
        logger=wandb_logger if not args.no_wandb else None,
        accelerator=device,
        devices=args.num_devices,
        log_every_n_steps=args.log_every_n_steps,
        precision=args.train_precision,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        use_distributed_sampler=args.num_devices > 1,
        val_check_interval=args.val_check_interval,
        limit_val_batches=0 if args.no_val else None,
        accumulate_grad_batches=accumulate_grad_batches,  # Key parameter
        gradient_clip_val=1.0,  # Prevent gradient explosion
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=False,  # Save some memory
        sync_batchnorm=False if args.num_devices == 1 else True
    )
    
    return trainer


# Usage example for modified training script:
"""
from gradient_accumulation import GradientAccumulationStrategy, create_memory_efficient_trainer

# Modify args for gradient accumulation
args, accumulate_steps = GradientAccumulationStrategy.modify_trainer_args(args, accumulate_grad_batches=8)

# Create memory-efficient trainer
trainer = create_memory_efficient_trainer(args, callbacks, wandb_logger, accumulate_steps)
"""
