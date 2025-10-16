# Axis 1: Representation Probing - Simple Start

## Quick Overview

This is a **lightweight, pragmatic** approach to probing the DreaMS model:

1. Work with **final embeddings only** (not all layers yet)
2. Test what's encoded using simple probes
3. Once it works → expand to per-layer analysis

## What You Have

```
notebooks/
├── dataset_creation.ipynb        # ✅ Done - enriched dataset ready
└── axis1_probing_simple.ipynb    # 📝 New - simple probing workflow

src/
└── simple_probing.py             # Helper functions for probing

data/processed/
└── MassSpecGym_enriched.tsv      # ✅ Ready with all targets
```

## Quick Start

### 1. Get Final Embeddings

You need embeddings from the DreaMS model (final layer output):

```python
# Shape: (n_spectra, embedding_dim)
# e.g., (231104, 1024) if embedding_dim=1024
```

**Options:**
- Load from saved file if you have them
- Extract from DreaMS model and save

### 2. Run Probing

Open `notebooks/axis1_probing_simple.ipynb` and run through the cells:

1. Load embeddings
2. Load targets (from enriched dataset)
3. Train linear probes
4. Train MLP probes
5. Compare results
6. Validate with kNN and UMAP

### 3. Answer Key Questions

The notebook will help you answer:

- **What's encoded?** Which properties (MW, LogP, aromatic, etc.) can be predicted?
- **Linear or not?** Do MLPs significantly outperform linear probes?
- **Validated?** Do kNN and UMAP agree with probe findings?

## Example Results

After running, you'll see outputs like:

```
aromatic (linear probe):
  AUROC: 0.873
  AP: 0.891
  Accuracy: 0.812

mol_weight (linear probe):
  R²: 0.654
  RMSE: 0.421
```

## Next Steps

Once this works:
- ✅ Understand what's in final embeddings
- ✅ Validate the approach
- → **Then** expand to per-layer analysis to see WHERE information emerges

## Files

- `notebooks/axis1_probing_simple.ipynb` - Main workflow
- `src/simple_probing.py` - Probing utilities
- `PROBING_APPROACH.md` - Why this approach

## Tips

1. **Start small**: Test with 1-2 targets first
2. **Check shapes**: Make sure embeddings and targets align
3. **Save results**: Results will be saved to `../results/`
4. **Iterate**: If something doesn't work, easy to debug with this simple setup

## Later: Per-Layer Analysis

Once you're happy with final embeddings, we can expand to:
- Extract embeddings from each layer
- Probe layer-by-layer
- See where information emerges (early vs late layers)
- Plot layer-wise curves

But for now, keep it simple! 🚀
