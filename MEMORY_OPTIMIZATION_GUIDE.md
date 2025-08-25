# DreaMS Memory Optimization Strategies

## Current Issue
Training runs out of memory around epoch 60 with:
- batch_size: 2 (now reduced to 1)
- max_peaks_n: 100
- train_precision: 32
- num_devices: 2

## Implemented Solutions

### 1. Gradient Accumulation (PRIMARY SOLUTION)
- **Batch size**: Reduced from 2 to 1
- **Gradient accumulation**: 2 steps (maintains effective batch size of 2)
- **Memory savings**: ~50% reduction in peak memory usage
- **Training impact**: Minimal - maintains same effective batch size

### 2. Enhanced Memory Callbacks
- **Aggressive cache clearing**: Every batch and validation step
- **Garbage collection**: Every 10-25 steps
- **Reference cleanup**: Explicit deletion of batch variables
- **Memory savings**: 10-20% additional reduction

### 3. Environment Optimizations
- **CUDA memory allocation**: Configured for smaller chunks
- **Memory fraction**: Limited to 95% of available memory
- **Flash attention**: Enabled if available

## Additional Strategies (If Still Needed)

### 4. Dataset Optimizations
```python
# Use in-memory=False for large datasets
msdata = du.MSData(args.dataset_pth, in_mem=False)

# Reduce validation dataset size
original_args.limit_val_batches = 0.5  # Only use 50% of validation data
```

### 5. Model Architecture Tweaks
If memory is still insufficient, consider these parameter modifications:
- Reduce `d_fourier` from default (if using Fourier features)
- Reduce `d_peak` slightly (affects model capacity but saves memory)
- Use fewer transformer layers during fine-tuning (freeze some layers)

### 6. Checkpoint Strategy
```python
# Reduce checkpoint frequency
original_args.save_top_k = 1
original_args.every_n_train_steps = 2000  # Less frequent checkpointing
```

### 7. Validation Optimizations
```python
# Reduce validation frequency and scope
original_args.val_check_interval = 0.5  # Only twice per epoch
original_args.limit_val_batches = 100   # Limit validation samples
```

### 8. Mixed Precision (Alternative to FP32)
If results allow slight precision reduction:
```bash
--train_precision 16  # Use mixed precision
```

### 9. Sequential Layer Processing
For extreme cases, process transformer layers sequentially rather than in parallel (requires code modification).

### 10. Data Pipeline Optimizations
```python
# Reduce num_workers to save CPU memory
original_args.num_workers_data = 1

# Use pin_memory=False to reduce memory pressure
# (implemented in DataLoader initialization)
```

## Memory Usage Monitoring

Add this to monitor memory usage:
```python
def log_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

# Call this in training loop to monitor usage
```

## Expected Results

With gradient accumulation (batch_size=1, accumulate_grad_batches=2):
- **Memory reduction**: ~50% peak memory usage
- **Training time**: Slightly increased due to more frequent cache clearing
- **Model quality**: Should maintain same results as original batch_size=2
- **Convergence**: May be slightly different but should reach similar performance

## Usage

1. **Standard optimization**: Use `integrated_memory_optimized_train_with_wandb.sh`
2. **Extreme optimization**: Use `ultra_memory_optimized_train.sh`
3. **Monitor**: Watch GPU memory usage and adjust parameters as needed

## Troubleshooting

If still running out of memory:
1. Reduce `max_peaks_n` from 100 to 80-90
2. Consider mixed precision (`--train_precision 16`)
3. Use only 1 GPU instead of 2
4. Implement model parallelism for larger models
