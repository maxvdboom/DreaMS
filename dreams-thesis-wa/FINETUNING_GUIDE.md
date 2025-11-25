# MassSpecGym Fine-Tuning Workflow

This directory contains scripts for fine-tuning DreaMS on the MassSpecGym dataset with Morgan fingerprint prediction.

## Overview

The workflow consists of two steps:
1. **Data Preparation**: Convert `MassSpecGym_enriched.tsv` to HDF5 format
2. **Fine-Tuning**: Train the model to predict Morgan 2048 fingerprints

## Step 1: Prepare Dataset

Convert the enriched TSV file to HDF5 format suitable for DreaMS fine-tuning:

```bash
cd dreams-thesis-wa
python src/prepare_massspecgym_for_finetuning.py
```

**What it does:**
- Reads `data/raw/MassSpecGym.tsv`
- Extracts essential columns:
  - `identifier`, `mzs`, `intensities`, `smiles`, `inchikey`
  - Metadata: `precursor_mz`, `adduct`, `instrument_type`, `collision_energy`, `fold`
- Processes spectra (max 128 peaks, intensity-normalized)
- Creates `data/processed/MassSpecGym_finetuning.hdf5`

**Output:**
```
MassSpecGym_finetuning.hdf5
├── spectrum (N, 2, 128)      # m/z and intensity arrays
├── precursor_mz (N,)
├── charge (N,)
├── adduct (N,)
├── smiles (N,)               # Required for Morgan FP generation
├── identifier (N,)
├── inchikey (N,)
└── [other metadata...]
```

## Step 2: Fine-Tune Model

Run fine-tuning with Morgan 2048 fingerprints:

```bash
cd .. # Back to DreaMS root

# Option 1: Edit the script to set your WandB project
# Edit dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh
# Change: WANDB_PROJECT="your-wandb-project-name"

bash dreams-thesis-wa/scripts/finetune_massspecgym_morgan2048.sh
```

**WandB Integration:**
- Set `WANDB_PROJECT="your-project-name"` in the script to enable WandB logging
- Optionally set `WANDB_ENTITY="your-username"` for team projects
- If not configured, will run with `--no_wandb` (local logging only)

**What it does:**
- Loads pre-trained DreaMS model from `${PRETRAINED}/ssl_model.ckpt`
- Fine-tunes on `MassSpecGym_finetuning.hdf5`
- Training objective: `fp_morgan_2048` (2048-bit Morgan fingerprints)
- Saves checkpoints to `lightning_logs/`

**Hyperparameters:**
- Learning rate: 1e-4
- Batch size: 32
- Max epochs: 50
- Validation fraction: 10%
- Head depth: 2 layers
- Max peaks: 128

## Requirements

- Pre-trained DreaMS model (SSL checkpoint)
- `MassSpecGym.tsv` in `data/raw/`
- DreaMS environment with all dependencies

## Output

After fine-tuning, you'll have:
- Model checkpoints in `lightning_logs/MassSpecGym_Morgan2048/`
- Training metrics and logs
- Top-3 best models (by validation loss)

## Customization

### Adjust peak limit:
Edit `prepare_massspecgym_for_finetuning.py`:
```python
convert_tsv_to_hdf5(tsv_path, output_path, n_highest_peaks=256)  # Default: 128
```

### Change batch size or learning rate:
Edit `finetune_massspecgym_morgan2048.sh`:
```bash
--batch_size 64 \      # Default: 32
--lr 5e-5 \            # Default: 1e-4
```

### Use Morgan 4096 instead:
```bash
--train_objective fp_morgan_4096 \
```

## Troubleshooting

**Dataset not found:**
```
Error: Dataset not found at ./dreams-thesis-wa/data/processed/MassSpecGym_finetuning.hdf5
```
→ Run Step 1 first: `python src/prepare_massspecgym_for_finetuning.py`

**Pre-trained model not found:**
```
Error: Pre-trained model not found at ${PRETRAINED}/ssl_model.ckpt
```
→ Set environment variable: `export PRETRAINED=/path/to/pretrained/model/dir`

**Out of memory:**
→ Reduce batch size: `--batch_size 16` or `--batch_size 8`
