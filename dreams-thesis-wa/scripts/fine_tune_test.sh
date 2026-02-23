#!/bin/bash
#SBATCH --job-name=DreaMS_fine-tuning
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00

# Loading modules
module load 2024
module load Miniconda3/24.7.1-0

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate dreams

# Export project definitions
$(python -c "from dreams.definitions import export; export()")

# Set up scratch directory
SCRATCH_DIR="/scratch-shared/$USER/dreams_finetune_$$"
HOME_OUTPUT_DIR="$HOME/DreaMS/dreams-thesis-wa/results/finetuning"

echo "Setting up scratch-shared workspace..."
mkdir -p "$SCRATCH_DIR"
mkdir -p "$HOME_OUTPUT_DIR"

# Copy input files to scratch (dataset + pre-trained model)
echo "Copying dataset to scratch..."
cp "$HOME/DreaMS/dreams-thesis-wa/data/processed/finetuning.hdf5" "$SCRATCH_DIR/"
echo "Copying pre-trained model to scratch..."
cp "${PRETRAINED}/ssl_model.ckpt" "$SCRATCH_DIR/"

# Set paths to scratch locations
SCRATCH_DATASET="$SCRATCH_DIR/finetuning.hdf5"
SCRATCH_PRETRAINED="$SCRATCH_DIR/ssl_model.ckpt"
SCRATCH_CHECKPOINTS="$SCRATCH_DIR/checkpoints"
mkdir -p "$SCRATCH_CHECKPOINTS"

echo "✅ Scratch setup complete: $SCRATCH_DIR"

# Configuration
PROJECT_NAME="MassSpecGym_Morgan2048"
RUN_NAME="massspecgym_morgan2048_finetune_$(date +%Y%m%d_%H%M%S)"
DATASET_PATH="$SCRATCH_DATASET"

# WandB configuration
# Credentials are stored in .wandb_secrets (gitignored)gi
# See .wandb_secrets.template for setup instructions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.wandb_secrets" ]; then
    source "$SCRIPT_DIR/.wandb_secrets"
    echo "✅ Loaded WandB credentials from .wandb_secrets"
elif [ -f "$HOME/.wandb_secrets" ]; then
    source "$HOME/.wandb_secrets"
    echo "✅ Loaded WandB credentials from ~/.wandb_secrets"
elif [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  Warning: No WandB credentials found!"
    echo "   Create .wandb_secrets from .wandb_secrets.template"
    echo "   Or set WANDB_API_KEY environment variable"
fi
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

# Move to scratch directory so checkpoints are written to fast storage
cd "$SCRATCH_DIR" || exit 3

# We need the DreaMS package, so add it to PYTHONPATH
export PYTHONPATH="$HOME/DreaMS:$PYTHONPATH"

# Run the training script with srun for SLURM
srun --export=ALL --preserve-env python3 "$HOME/DreaMS/dreams/training/train.py" \
 $WANDB_ARGS \
 --job_key "$RUN_NAME" \
 --run_name "$RUN_NAME" \
 --train_objective fp_morgan_2048 \
 --train_regime fine-tuning \
 --dataset_pth "$DATASET_PATH" \
 --dformat A \
 --model DreaMS \
 --lr 7.5e-6 \
 --batch_size 512 \
 --prec_intens 1.1 \
 --num_devices 4 \
 --max_epochs 103 \
 --log_every_n_steps 5 \
 --head_depth 1 \
 --seed 3407 \
 --train_precision 32 \
 --pre_trained_pth "$SCRATCH_PRETRAINED" \
 --val_check_interval 0.1 \
 --max_peaks_n 100 \
 --save_top_k 3 \
 --num_workers 32


# Zipping checkpoints and copying to home
# Checkpoints are saved to {project_name}/{job_key}/ in the working directory (scratch)
CHECKPOINT_DIR="$SCRATCH_DIR/$WANDB_PROJECT/$RUN_NAME"
echo ""
echo "Zipping checkpoints from $CHECKPOINT_DIR..."
if [ -d "$CHECKPOINT_DIR" ]; then
    cd "$SCRATCH_DIR"
    zip -r "${RUN_NAME}_checkpoints.zip" "$WANDB_PROJECT/$RUN_NAME/"
    echo "✅ Created ${RUN_NAME}_checkpoints.zip"
    
    echo "Copying zip to home directory..."
    mv "${RUN_NAME}_checkpoints.zip" "$HOME_OUTPUT_DIR/"
    echo "✅ Checkpoints saved to: $HOME_OUTPUT_DIR/${RUN_NAME}_checkpoints.zip"
else
    echo "⚠️ Warning: Checkpoint directory not found at $CHECKPOINT_DIR"
    echo "   Listing scratch contents:"
    ls -la "$SCRATCH_DIR"
fi

# Clean up scratch
echo "Cleaning up scratch directory..."
rm -rf "$SCRATCH_DIR"
echo "✅ Scratch cleaned up"

echo ""
echo "=================================="
echo "Fine-tuning complete!"
echo "Output: $HOME_OUTPUT_DIR/${RUN_NAME}_checkpoints.zip"
echo "=================================="

# Contrastive fine-tuning (commented out)
# python3 training/train.py \
#  --project_name CONTRASTIVE_FINE_TUNING \
#  --job_key "lr5e-6_margin0.1_fixed_rel_intens_max_peaks_n100" \
#  --run_name "lr5e-6_margin0.1_fixed_rel_intens_max_peaks_n100" \
#  --train_objective contrastive_spec_embs \
#  --train_regime fine-tuning \
#  --dformat A \
#  --model DreaMS \
#  --lr 5e-6 \
#  --batch_size 4 \
#  --prec_intens 1.1 \
#  --num_devices 8 \
#  --max_epochs 301 \
#  --log_every_n_steps 5 \
#  --seed 3407 \
#  --train_precision 32 \
#  --val_check_interval 1.0 \
#  --save_top_k -1 \
#  --head_depth 0 \
#  --unfreeze_backbone_at_epoch 0 \
#  --dataset_pth "${MERGED_DATASETS}/MoNA_A_Murcko_split_neighbours_[M+H]+_0.05Da.pkl" \
#  --pre_trained_pth "${PRETRAINED}/ssl_model.ckpt" \
#  --n_pos_samples 1 \
#  --n_neg_samples 1 \
#  --triplet_loss_margin 0.1 \
#  --max_peaks_n 100
