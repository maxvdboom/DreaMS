#!/bin/bash
# Enhanced memory-optimized fine-tuning script with WandB support

# WandB Configuration Options
# Choose ONE of the following options:

# Option 1: Set your API key directly (RECOMMENDED for cluster use)
# Get your API key from: https://wandb.ai/authorize
# export WANDB_API_KEY="your_api_key_here"

# Option 2: Use offline mode if you have connection issues
# export WANDB_MODE=offline

# Option 3: Use your personal WandB entity
export WANDB_ENTITY="m-van-den-boom"

# ========================================
# IMPORTANT: Set your API key here!
# Replace with your actual API key from https://wandb.ai/authorize
# ========================================
export WANDB_API_KEY="122c488a229c45237791eb8cb419d1bc2ecc577a"

# Uncomment the line below if you want to use offline mode instead
# export WANDB_MODE=offline

# First, set up environment variables
$(python -c "from dreams.definitions import export; export()")

# Check WandB status
echo "Checking WandB configuration..."

if [ "$WANDB_MODE" = "offline" ]; then
    echo "✓ WandB set to offline mode"
elif [ ! -z "$WANDB_API_KEY" ] && [ "$WANDB_API_KEY" != "REPLACE_WITH_YOUR_API_KEY" ]; then
    echo "✓ WandB API key is set"
else
    echo "⚠ WandB configuration incomplete!"
    echo ""
    echo "Please choose ONE of the following options:"
    echo ""
    echo "Option 1: Set your API key in the script"
    echo "  1. Get your API key from: https://wandb.ai/authorize"
    echo "  2. Replace 'REPLACE_WITH_YOUR_API_KEY' in this script with your actual API key"
    echo ""
    echo "Option 2: Use offline mode"
    echo "  Uncomment the line: export WANDB_MODE=offline"
    echo ""
    echo "Option 3: Set environment variable before running"
    echo "  export WANDB_API_KEY=your_api_key_here"
    echo "  bash integrated_memory_optimized_train_with_wandb.sh"
    echo ""
fi

if command -v wandb &> /dev/null; then
    echo "✓ WandB package is installed"
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
 --wandb_entity_name "${WANDB_ENTITY:-m-van-den-boom}" \
 --train_objective fp_morgan_4096 \
 --train_regime fine-tuning \
 --dataset_pth "./data/paired_spectra/canopus_train_dreams.hdf5" \
 --dformat A \
 --model DreaMS \
 --num_workers_data 4 \
 --lr 1e-4 \
 --batch_size 6 \
 --prec_intens 1.1 \
 --num_devices 2 \
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
