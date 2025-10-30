# DreaMS Thesis - Embedding Analysis

> Experimental framework for analyzing DreaMS embeddings and fine-tuning them for downstream machine learning tasks.

## Overview

This repository contains experiments with **DreaMS** (Deep Representations for Mass Spectrometry) embeddings for molecular property prediction and other downstream tasks. The work focuses on extracting and evaluating embeddings from pre-trained models for use in various machine learning applications.

## Project Structure

```
dreams-thesis-wa/
├── notebooks/              # Analysis and experimentation notebooks
│   ├── compare_ssl_vs_contrastive_embeddings.ipynb
│   └── dataset_creation.ipynb
├── src/                    # Utility scripts
│   ├── generate_ssl_embeddings.py
│   ├── simple_probing.py
│   ├── add_embeddings_to_tsv.py
│   ├── convert_parquet_to_hdf5.py
│   └── scaffold_splits.py
├── data/                   # Data files (not tracked)
│   ├── processed/
│   └── raw/
├── results/                # Experimental results
└── requirements.txt        # Python dependencies

---

**Part of the DreaMS framework** | [Main Repository](https://github.com/maxvdboom/DreaMS)