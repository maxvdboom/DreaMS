#!/bin/bash
# Enhanced memory-optimized fine-tuning script with WandB support

# WandB Configuration Options
# Uncomment and modify as needed:

# Option 1: Use your personal WandB entity (recommended)
# export WANDB_ENTITY="your_username"

# Option 2: Use offline mode if you have connection issues
# export WANDB_MODE=offline

# Option 3: Set API key directly (not recommended for security)
# export WANDB_API_KEY="your_api_key_here"

# Option 4: Use existing team entity
export WANDB_ENTITY="mass-spec-ml"

# First, set up environment variables
$(python -c "from dreams.definitions import export; export()")

# Check WandB status
echo "Checking WandB configuration..."
if command -v wandb &> /dev/null; then
    if wandb whoami &> /dev/null; then
        echo "✓ WandB is configured and logged in as: $(wandb whoami)"
    else
        echo "⚠ WandB is installed but not logged in. Please run:"
        echo "  wandb login"
        echo "  Or set WANDB_API_KEY environment variable"
        echo "  Or use WANDB_MODE=offline for offline logging"
    fi
else
    echo "⚠ WandB not found. Installing..."
    pip install wandb
fi

# Run the memory-optimized fine-tuning with WandB enabled
echo "Starting memory-optimized fine-tuning with WandB logging..."
python3 dreams/training/train.py \
 --project_name MorganFingerprints \
 --job_key "morgan_4096_fine_tune_wandb" \
 --run_name "morgan_4096_fine_tune_wandb_$(date +%Y%m%d_%H%M%S)" \
 --wandb_entity_name "${WANDB_ENTITY:-mass-spec-ml}" \
 --train_objective fp_morgan_4096 \
 --train_regime fine-tuning \
 --dataset_pth "./data/paired_spectra/canopus_train_dreams.hdf5" \
 --dformat A \
 --model DreaMS \
 --num_workers_data 4 \
 --lr 1e-4 \
 --batch_size 2 \
 --prec_intens 1.1 \
 --num_devices 1 \
 --max_epochs 100 \
 --log_every_n_steps 10 \
 --head_depth 1 \
 --seed 3407 \
 --train_precision 32 \
 --pre_trained_pth "${PRETRAINED}/ssl_model.ckpt" \
 --val_check_interval 0.1 \
 --max_peaks_n 100 \
 --save_top_k 3 \
 --val_frac 0.2 \
 --weight_decay 1e-5

echo "Training completed. Check your WandB dashboard for logs and metrics."
