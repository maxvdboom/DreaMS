#!/bin/bash
"""
Fine-tuning script for MassSpecGym dataset with Morgan 2048 fingerprints.

This script fine-tunes a pre-trained DreaMS model on the MassSpecGym dataset
to predict Morgan fingerprints (2048-bit).

Prerequisites:
1. Run prepare_massspecgym_for_finetuning.py to create the HDF5 file
2. Have a pre-trained DreaMS model checkpoint

Usage:
    bash finetune_massspecgym_morgan2048.sh
"""

# Set up environment variables
$(python -c "from dreams.definitions import export; export()")

# Configuration
PROJECT_NAME="MassSpecGym_Morgan2048"
RUN_NAME="massspecgym_morgan2048_finetune_$(date +%Y%m%d_%H%M%S)"
DATASET_PATH="./dreams-thesis-wa/data/processed/MassSpecGym_finetuning.hdf5"

# WandB configuration
# Get your API key from: https://wandb.ai/authorize
export WANDB_API_KEY="122c488a229c45237791eb8cb419d1bc2ecc577a"
export WANDB_ENTITY="SSL_MAC"
WANDB_PROJECT="MorganFingerprints"  # Your WandB project name

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset not found at $DATASET_PATH"
    echo "Please run: python dreams-thesis-wa/src/prepare_massspecgym_for_finetuning.py"
    exit 1
fi

# Check if pre-trained model exists
if [ ! -f "${PRETRAINED}/ssl_model.ckpt" ]; then
    echo "Error: Pre-trained model not found at ${PRETRAINED}/ssl_model.ckpt"
    echo "Please set PRETRAINED environment variable to your pre-trained model directory"
    exit 1
fi

echo "=================================="
echo "MassSpecGym Fine-Tuning"
echo "=================================="
echo "Project: $PROJECT_NAME"
echo "Run: $RUN_NAME"
echo "Dataset: $DATASET_PATH"
echo "Objective: Morgan Fingerprints (2048-bit)"
echo "Pre-trained model: ${PRETRAINED}/ssl_model.ckpt"
echo "WandB Project: $WANDB_PROJECT"
echo "=================================="
echo ""

# Build WandB arguments
WANDB_ARGS=""
if [ ! -z "$WANDB_PROJECT" ] && [ "$WANDB_PROJECT" != "your-wandb-project-name" ]; then
    WANDB_ARGS="--project_name $WANDB_PROJECT --wandb_entity_name $WANDB_ENTITY"
    echo "WandB logging enabled"
else
    WANDB_ARGS="--no_wandb"
    echo "WandB logging disabled (set WANDB_PROJECT to enable)"
fi
echo ""

# Fine-tuning with Morgan 2048 fingerprints
python3 dreams/training/train.py \
 $WANDB_ARGS \
 --job_key "$RUN_NAME" \
 --run_name "$RUN_NAME" \
 --train_objective fp_morgan_2048 \
 --train_regime fine-tuning \
 --dataset_pth "$DATASET_PATH" \
 --dformat A \
 --model DreaMS \
 --num_workers_data 4 \
 --lr 1e-4 \
 --batch_size 64 \
 --prec_intens 1.1 \
 --num_devices 1 \
 --max_epochs 100 \
 --log_every_n_steps 50 \
 --head_depth 2 \
 --seed 3407 \
 --train_precision 64 \
 --pre_trained_pth "${PRETRAINED}/ssl_model.ckpt" \
 --val_check_interval 0.25 \
 --max_peaks_n 128 \
 --save_top_k 3 \
 --val_frac 0.1 \
 --weight_decay 1e-5

echo ""
echo "=================================="
echo "Fine-tuning completed!"
echo "=================================="
