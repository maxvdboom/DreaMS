# DreaMS Thesis - Embedding Analysis

> Experimental framework for analyzing DreaMS embeddings and fine-tuning them for downstream machine learning tasks.

## Overview

This repository contains experiments with **DreaMS** (Deep Representations for Mass Spectrometry) embeddings for molecular property prediction and other downstream tasks. The work focuses on extracting and evaluating embeddings from pre-trained models for use in various machine learning applications.

## Project Structure

```
dreams-thesis-wa/
├── src/                    # Core data processing pipeline
│   ├── README.md                          # Pipeline documentation
│   ├── generate_ssl_embeddings.py         # Step 1: Generate SSL embeddings
│   ├── add_rdkit_descriptors.py           # Step 2: Add molecular descriptors
│   ├── murcko_histogram_splits.py         # Step 3: Create train/val/test splits
│   ├── prepare_massbank_external_test.py  # Step 4: Prepare external validation
│   └── compute_massbank_embeddings.py     # Step 5: External SSL embeddings
│
├── notebooks/              # Analysis notebooks
│   ├── probe_ssl_embeddings.ipynb         # Main: Internal validation (probing)
│   ├── external_validation_massbank.ipynb # Main: External validation
│   └── exploratory/                       # Archived/exploratory notebooks
│       └── dataset_creation.ipynb         # Reference only
│
├── data/                   # Data files (not tracked in git)
│   ├── raw/                               # Original MassSpecGym.tsv
│   ├── processed/                         # Processed parquet files with embeddings
│   └── external/                          # MassBank EU external validation set
│
├── results/                # Experimental results
│   ├── probing_results_ssl.pkl
│
├── figures/                # Created figures
│   ├── ssl_embedding_baseline_linear_vs_mlp.png
│
├── models/                 # Pre-trained model checkpoints
│   └── ssl_model.ckpt
│
└── requirements.txt        # Python dependencies
```

## Quick Start

See detailed pipeline documentation in **[`src/README.md`](src/README.md)**

**Run the full pipeline:**
```bash
# Step 1: Generate SSL embeddings (1024-dim from DreaMS model)
python src/generate_ssl_embeddings.py

# Step 2: Add 10 RDKit molecular descriptors
python src/add_rdkit_descriptors.py

# Step 3: Create rigorous Murcko histogram splits (train/val/test)
python src/murcko_histogram_splits.py

# Step 4-5 (Optional): Prepare external validation set (MassBank EU)
python src/prepare_massbank_external_test.py
python src/compute_massbank_embeddings.py

# Step 6: Run analysis notebooks
# - notebooks/probe_ssl_embeddings.ipynb (internal validation)
# - notebooks/external_validation_massbank.ipynb (external validation)
```

---

**Part of the DreaMS framework** | [Main Repository](https://github.com/maxvdboom/DreaMS)