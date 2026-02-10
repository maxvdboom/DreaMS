# Indicator Analysis Pipeline

**Overview**: Three-stage pipeline to compute SSL embedding quality indicators using nearest-neighbor analysis.

## Data Flow

```
TASK 0 (Setup)
    ↓ [reads probing_test.parquet + all_rdkit_descriptors.parquet]
    ↓ [normalizes embeddings, merges on 'smiles']
    ↓ [saves indicator_data.pkl]
    ↓
    ├── indicator_data.pkl  (48 MB)
    │   ├── embeddings: (45,185 × 1024)
    │   ├── descriptors: (45,185 × 208)
    │   ├── descriptor_names: list of 208 names
    │   ├── spectrum_to_molecule: array of InChIKeys
    │   ├── molecule_to_spectra: dict InChIKey → spectrum indices
    │   └── ... [metadata]
    │
    ├──────────────────────────────────────────┐
    │                                          │
    TASK 1 (Build kNN Graphs)          [cached in memory]
        ↓ [reads indicator_data.pkl]
        ↓ [computes cosine-distance kNN for k ∈ {10, 50, 100}]
        ↓ [filters exclusive version: removes same-molecule neighbors]
        ↓ [saves 6 kNN graphs]
        │
        ├── knn_k10_inclusive.pkl  (2.1 MB)  ← indices, distances, similarities
        ├── knn_k10_exclusive.pkl  (2.1 MB)
        ├── knn_k50_inclusive.pkl  (10.5 MB)
        ├── knn_k50_exclusive.pkl  (10.5 MB)
        ├── knn_k100_inclusive.pkl (21 MB)
        └── knn_k100_exclusive.pkl (21 MB)
        │
        └──────────────────────────────────────────┐
                                                   │
        TASK 2 (Indicator 1: NN Descriptor Consistency)
            ↓ [reads indicator_data.pkl]
            ↓ [reads knn_k{10,50,100}_{inclusive,exclusive}.pkl]
            ↓ [computes for each descriptor:]
            ↓   - mean |Δdescriptor| for NN pairs
            ↓   - mean |Δdescriptor| for random pairs
            ↓   - effect_ratio = random_mean / nn_mean
            ↓   - Spearman(embedding_similarity, -|Δdescriptor|)
            │
            └── nn_descriptor_consistency.csv
                ├── Rows: 208 descriptors × 3 k values × 2 neighbor types
                ├── Columns: descriptor, k, neighbors, nn_mean_diff, 
                │            random_mean_diff, effect_ratio, spearman_corr
                └── Summary: Top/Bottom 10 by effect_ratio (k=50, exclusive)

        TASK 3 (Indicator 2: Clustering Purity)              [future]
            ↓ [reads indicator_data.pkl + knn graphs]
            │
            └── clustering_purity.csv

        TASK 4 (Indicator 3: Structural Separation)          [future]
            ↓ [reads indicator_data.pkl + knn graphs]
            │
            └── structural_separation.csv

        TASK 5 (Integration & Figures)                       [future]
            ↓ [reads all indicator CSVs]
            │
            └── integrated_indicator_analysis.csv
                + figures/
                  ├── descriptor_consistency_heatmap.pdf
                  ├── clustering_purity_scatter.pdf
                  └── ...
```

## Key Cached Artifacts

### 1. `indicator_data.pkl` (TASK 0)
- **Size**: ~48 MB
- **Contents**:
  - `embeddings`: (45,185, 1024) float32 array — SSL embeddings
  - `descriptors`: (45,185, 208) float32 array — RDKit descriptors
  - `descriptor_names`: list of 208 descriptor names
  - `spectrum_to_molecule`: (45,185,) array of InChIKeys
  - `molecule_to_spectra`: dict mapping InChIKey → list of spectrum indices
  - `inchikeys`, `smiles`, metadata (n_spectra, n_molecules, etc.)

### 2. kNN Graphs (TASK 1)
**6 files total **:
- **Inclusive** (k=10, 50, 100): All k nearest neighbors
  - `knn_k{k}_inclusive.pkl`: {indices, distances, similarities, metadata}
- **Exclusive** (k=10, 50, 100): Neighbors from different molecules only
  - `knn_k{k}_exclusive.pkl`: {indices (with -1 padding), distances (NaN padded), similarities, metadata}

**Sizes**:
- k=10: ~2.1 MB each (inclusive + exclusive)
- k=50: ~10.5 MB each
- k=100: ~21 MB each

### 3. Results CSV Files

#### `nn_descriptor_consistency.csv` (TASK 2)
- **Rows**: 208 descriptors × 3 k values × 2 neighbor types = 1,248 rows
- **Columns**:
  - `descriptor`: descriptor name
  - `k`: neighborhood size (10, 50, 100)
  - `neighbors`: 'inclusive' or 'exclusive'
  - `nn_mean_diff`: mean |Δdescriptor| in kNN
  - `random_mean_diff`: mean |Δdescriptor| in random pairs
  - `effect_ratio`: random_mean / nn_mean (higher = more consistent)
  - `spearman_corr`: correlation between embedding similarity and descriptor agreement

## Execution SEquence

```bash
# First run setup
jupyter notebook task0_indicator_setup.ipynb
# Run all cells (TASK 0)

# Then build kNN graphs once
jupyter notebook task1_build_knn_graphs.ipynb
# Run all cells (TASK 1)

# Run indicator analyses in any order
jupyter notebook task2_indicator1_nn_consistency.ipynb
# jupyter notebook task3_indicator2_clustering_purity.ipynb  [future]
# jupyter notebook task4_indicator3_structural_separation.ipynb  [future]

# Finally integrate and visualize
# jupyter notebook task5_integrate_indicators.ipynb  [future]
```

## Key Design Decisions

1. **Separate kNN computation** (TASK 1):
   - Compute once, reuse across all descriptors
   - Avoid redundant cosine distance calculations

2. **Two neighbor versions**:
   - **Inclusive**: all k neighbors (sanity check)
   - **Exclusive**: filter same-molecule neighbors (real analysis)
   - Both stored for comparison

3. **Effect ratio = random/NN**:
   - Ratio > 1: descriptor is consistent in embedding space
   - Ratio < 1: descriptor is inconsistent

4. **Spearman correlation**:
   - Correlates embedding similarity with descriptor agreement
   - Validates that geometry captures descriptor structure

## Validation Checklist

- [ ] TASK 0: All 45,185 spectra load correctly
- [ ] TASK 0: All 208 descriptors present (no NaN values > threshold)
- [ ] TASK 0: indicator_data.pkl saved (~48 MB)
- [ ] TASK 1: 6 kNN graph caches saved (~68 MB total)
- [ ] TASK 1: Exclusive graphs have valid-neighbor statistics logged
- [ ] TASK 2: CSV has 1,248 rows (208 × 3 × 2)
- [ ] TASK 2: effect_ratio values are positive (random_mean >> 0)
- [ ] TASK 2: Top 10 descriptors have ratio > 1 (consistent)
- [ ] TASK 2: Summary printed to console
