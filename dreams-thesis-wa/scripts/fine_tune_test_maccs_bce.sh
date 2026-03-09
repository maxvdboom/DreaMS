#!/bin/bash
#SBATCH --job-name=DreaMS_ft_maccs_bce
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00

# MACCS 166 + BCE loss
export FP_OBJECTIVE=fp_maccs_166
export FP_LOSS=bce_logits
export PROJECT_NAME=MassSpecGym_MACCS166_BCE
# export FP_POS_WEIGHT=6  # Uncomment for pos_weight (MACCS ~15% density)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/fine_tune_test.sh"
