#!/usr/bin/env python3
"""
Memory-optimized FingerprintHead that prevents validation output accumulation.
"""

import torch
import torch.nn as nn
from dreams.models.heads.heads import FingerprintHead as OriginalFingerprintHead


class MemoryOptimizedFingerprintHead(OriginalFingerprintHead):
    """
    Memory-optimized version of FingerprintHead that prevents memory leaks.
    """
    
    def __init__(self, *args, **kwargs):
        # Disable validation output storage by default
        kwargs['store_val_out_dir'] = None
        super().__init__(*args, **kwargs)
        
        # Reduce retrieval frequency to save memory
        self.retrieval_epoch_freq = max(20, self.retrieval_epoch_freq)
    
    def validation_step(self, data, batch_idx, dataloader_idx=0):
        """
        Memory-optimized validation step.
        """
        pred, loss = self.step(data, batch_idx)
        real = data['label']

        # Detach tensors to prevent gradient accumulation
        pred_detached = pred.detach()
        real_detached = real.detach()
        
        metrics = self.val_metrics(pred_detached, real_detached)
        metrics[f'Val loss'] = loss.detach()

        self.log_dict(metrics, sync_dist=True, on_epoch=True, on_step=False, 
                      batch_size=self.batch_size, add_dataloader_idx=False)

        # Skip memory-intensive retrieval validation more often
        if self.__retrieval_epoch() and batch_idx % 10 == 0:  # Only every 10th batch
            for i in range(len(pred_detached)):
                if hasattr(self, 'val_retrieval') and self.val_retrieval:
                    self.val_retrieval.retrieve_inchi14s(
                        query_fp=pred_detached[i].cpu().numpy(), 
                        label_smiles=data['smiles'][i]
                    )

        # Clear references
        del pred_detached, real_detached
        
        return loss.detach()
    
    def on_validation_epoch_end(self):
        """
        Memory-optimized validation epoch end.
        """
        if self.__retrieval_epoch():
            if hasattr(self, 'val_retrieval') and self.val_retrieval and self.val_retrieval.n_retrievals > 0:
                metrics_avg, _ = self.val_retrieval.compute_reset_metrics('Val', return_unaveraged=False)
                self.log_dict(metrics_avg, sync_dist=True, batch_size=self.batch_size)
        
        # Force garbage collection
        torch.cuda.empty_cache()


# Monkey patch the original class (use with caution)
def apply_memory_optimizations():
    """
    Apply memory optimizations by replacing the original FingerprintHead.
    """
    import dreams.models.heads.heads as heads_module
    heads_module.FingerprintHead = MemoryOptimizedFingerprintHead
    print("Applied memory optimizations to FingerprintHead")


# Usage:
# from memory_optimized_head import apply_memory_optimizations
# apply_memory_optimizations()  # Call before creating the model
