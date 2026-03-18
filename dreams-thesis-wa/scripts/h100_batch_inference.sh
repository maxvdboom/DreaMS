#!/bin/bash
#SBATCH --job-name=dreams-axis2-infer
#SBATCH --partition=gpu_h100
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

# Optional cluster/account override:
# SBATCH --account=<your_account>

set -euo pipefail

# ------------------------------------------------------------------
# Modules & conda (same pattern as fine-tune scripts)
# ------------------------------------------------------------------
module load 2024
module load Miniconda3/24.7.1-0

eval "$(conda shell.bash hook)"
conda activate dreams

# Export definitions so PRETRAINED and related paths are available.
$(python -c "from dreams.definitions import export; export()")

# ------------------------------------------------------------------
# Paths and run configuration
# ------------------------------------------------------------------
REPO_ROOT="$HOME/DreaMS"
SCRIPT_PATH="$REPO_ROOT/dreams-thesis-wa/scripts/h100_batch_inference.py"

# Persistent storage on cluster ($HOME)
PERSISTENT_CKPT_BASE_DIR="${CKPT_BASE_DIR:-}"
PERSISTENT_OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/dreams-thesis-wa/results/model_runs}"

# Fast node-local/shared scratch
RUN_LABEL="${RUN_LABEL:-all6}"
SCRATCH_ROOT="/scratch-shared/$USER/dreams_axis2_infer_${RUN_LABEL}_${SLURM_JOB_ID}"
SCRATCH_DATA_DIR="$SCRATCH_ROOT/data"
SCRATCH_CKPT_DIR="$SCRATCH_ROOT/checkpoints"
SCRATCH_OUTPUT_ROOT="$SCRATCH_ROOT/model_runs"

mkdir -p "$SCRATCH_DATA_DIR" "$SCRATCH_CKPT_DIR" "$SCRATCH_OUTPUT_ROOT" "$PERSISTENT_OUTPUT_ROOT"

echo ""
echo "=================================="
echo "DreaMS Axis2 H100 Inference"
echo "=================================="
echo "  Repo root            : $REPO_ROOT"
echo "  Script               : $SCRIPT_PATH"
echo "  Checkpoints (source) : $PERSISTENT_CKPT_BASE_DIR"
echo "  Persistent output    : $PERSISTENT_OUTPUT_ROOT"
echo "  Scratch root         : $SCRATCH_ROOT"
echo "=================================="
echo ""

if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Error: script not found at $SCRIPT_PATH"
  exit 1
fi

# Auto-detect checkpoint root if not explicitly provided.
if [ -z "$PERSISTENT_CKPT_BASE_DIR" ]; then
  for candidate in \
    "$HOME/THESIS/model_checkpoints" \
    "$HOME/DreaMS/dreams-thesis-wa/results/finetuning" \
    "$HOME/dreams-thesis-wa/results/finetuning"; do
    if [ -d "$candidate" ]; then
      PERSISTENT_CKPT_BASE_DIR="$candidate"
      break
    fi
  done
fi

if [ -z "$PERSISTENT_CKPT_BASE_DIR" ] || [ ! -d "$PERSISTENT_CKPT_BASE_DIR" ]; then
  echo "Error: checkpoint base dir not found."
  echo "Set it explicitly, e.g.:"
  echo "  sbatch --export=CKPT_BASE_DIR=/path/to/checkpoints dreams-thesis-wa/scripts/h100_batch_inference.sh"
  exit 1
fi

# ------------------------------------------------------------------
# Stage inputs to scratch
# ------------------------------------------------------------------
echo "Copying datasets to scratch..."
cp "$REPO_ROOT/dreams-thesis-wa/data/processed/MassSpecGym_splits/probing_test.parquet" "$SCRATCH_DATA_DIR/"
cp "$REPO_ROOT/dreams-thesis-wa/data/processed/MassSpecGym_splits/finetuning.hdf5" "$SCRATCH_DATA_DIR/"

echo "Copying checkpoints to scratch (recursive)..."
find "$PERSISTENT_CKPT_BASE_DIR" -type f -name "*.ckpt" -print0 | xargs -0 -I{} cp "{}" "$SCRATCH_CKPT_DIR/"

CKPT_COUNT=$(find "$SCRATCH_CKPT_DIR" -type f -name "*.ckpt" | wc -l | tr -d ' ')
if [ "$CKPT_COUNT" -eq 0 ]; then
  echo "Error: no .ckpt files found under $PERSISTENT_CKPT_BASE_DIR"
  exit 1
fi
echo "Copied $CKPT_COUNT checkpoint files to scratch."

# ------------------------------------------------------------------
# Run inference from scratch
# ------------------------------------------------------------------
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
cd "$SCRATCH_ROOT"

srun --export=ALL --preserve-env python "$SCRIPT_PATH" \
  --device cuda \
  --batch-size 1024 \
  --ckpt-base-dir "$SCRATCH_CKPT_DIR" \
  --probing-test "$SCRATCH_DATA_DIR/probing_test.parquet" \
  --finetuning-hdf5 "$SCRATCH_DATA_DIR/finetuning.hdf5" \
  --output-root "$SCRATCH_OUTPUT_ROOT" \
  ${RUN_TAGS:+--run-tags "$RUN_TAGS"}

# ------------------------------------------------------------------
# Sync per-run results back to persistent storage
# ------------------------------------------------------------------
echo ""
echo "Syncing run outputs to persistent folders..."
for run_dir in "$SCRATCH_OUTPUT_ROOT"/*; do
  [ -d "$run_dir" ] || continue
  run_tag="$(basename "$run_dir")"
  mkdir -p "$PERSISTENT_OUTPUT_ROOT/$run_tag"
  rsync -a "$run_dir/" "$PERSISTENT_OUTPUT_ROOT/$run_tag/"
  echo "  Synced: $run_tag -> $PERSISTENT_OUTPUT_ROOT/$run_tag"
done

echo ""
echo "Cleaning up scratch: $SCRATCH_ROOT"
rm -rf "$SCRATCH_ROOT"
echo "Done."

# Optional: run only specific run tags
# sbatch --export=RUN_TAGS="morgan_2048_cos,morgan_2048_bce" dreams-thesis-wa/scripts/h100_batch_inference.sbatch
