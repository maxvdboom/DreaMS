# DreaMS Thesis - Data Processing Pipeline

This directory contains the core data processing scripts for the DreaMS thesis project.

## 📋 Main Pipeline (Run in Order)

### 1. Generate SSL Embeddings
**Script:** `generate_ssl_embeddings.py`

Generates SSL (Self-Supervised Learning) embeddings for MassSpecGym spectra using the pre-trained DreaMS model.

```bash
python src/generate_ssl_embeddings.py
```

**Input:**
- Raw MassSpecGym data (TSV format)
- Pre-trained SSL model checkpoint (`ssl_model.ckpt`)

**Output:**
- `data/processed/massspecgym_complete/ssl_embs/MassSpecGym_with_SSL_embeddings.parquet`
- Contains: spectrum data + 1024-dim SSL embeddings

---

### 2. Add RDKit Molecular Descriptors
**Script:** `add_rdkit_descriptors.py`

Computes 10 molecular descriptors from SMILES strings and adds them to the dataset.

```bash
python src/add_rdkit_descriptors.py
```

**What it does:**
- ❌ Removes old descriptors: `mol_weight`, `logp`, functional groups (if present)
- ✅ Adds 10 new RDKit descriptors:
  - `alogp`: Wildman-Crippen LogP (lipophilicity)
  - `hba`: Hydrogen Bond Acceptors
  - `hbd`: Hydrogen Bond Donors
  - `tpsa`: Topological Polar Surface Area
  - `n_rotatable_bonds`: Number of Rotatable Bonds
  - `n_aromatic_rings`: Number of Aromatic Rings
  - `n_aliphatic_rings`: Number of Aliphatic Rings
  - `fsp3`: Fraction of sp³ carbons
  - `qed`: Quantitative Estimate of Drug-likeness
  - `sa_score`: Synthetic Accessibility Score

**Input:**
- `data/processed/massspecgym_complete/ssl_embs/MassSpecGym_with_SSL_embeddings.parquet`

**Output:**
- Same file (overwrites with updated descriptors)
- `data/processed/massspecgym_complete/ssl_embs/MassSpecGym_with_SSL_embeddings.parquet`

---

### 3. Create Murcko Histogram-Based Splits
**Script:** `murcko_histogram_splits.py`

Creates rigorous train/val/test splits that prevent data leakage by ensuring structurally similar molecules (not just identical scaffolds) don't appear across folds.

```bash
python src/murcko_histogram_splits.py
```

**Algorithm:**
- Groups molecules by Murcko histogram (scaffold representation)
- Uses `are_sub_hists(k=3, d=4)` to check similarity
- Iteratively assigns groups to test/val/train while preventing similar scaffolds across folds
- **More rigorous than GroupShuffleSplit** (prevents similar, not just identical)

**Input:**
- `data/processed/massspecgym_complete/ssl_embs/MassSpecGym_with_SSL_embeddings.parquet`

**Output:**
- `data/processed/massspecgym_complete/ssl_embs/MassSpecGym_with_SSL_embeddings_murcko_hist_splits.parquet`
- Adds `fold` column: `train` / `val` / `test`

**Split ratios:**
- Train: ~77% (159,271 spectra)
- Val: ~12% (26,648 spectra)
- Test: ~11% (45,185 spectra)

---

## 🧪 External Validation Pipeline

### 4. Prepare MassBank External Test Set
**Script:** `prepare_massbank_external_test.py`

Downloads and prepares an external validation set from MassBank EU.

```bash
python src/prepare_massbank_external_test.py
```

**What it does:**
- Downloads MassBank EU records from GitHub release
- Filters: Positive ESI, Orbitrap/QTOF instruments, CE 20-40 eV, ≥5 peaks
- Deduplicates against MassSpecGym by InChIKey (zero overlap)
- Computes 10 RDKit descriptors (same as step 2)

**Output:**
- `data/external/massbank_eu/massbank_eu_external_test_no_embeddings.parquet`
- 758 unique compounds

---

### 5. Compute MassBank SSL Embeddings
**Script:** `compute_massbank_embeddings.py`

Generates SSL embeddings for the MassBank external test set.

```bash
python src/compute_massbank_embeddings.py
```

**Input:**
- `data/external/massbank_eu/massbank_eu_external_test_no_embeddings.parquet`
- Pre-trained SSL model checkpoint

**Output:**
- `data/external/massbank_eu/massbank_eu_external_test.parquet`
- Contains: MassBank data + 1024-dim SSL embeddings

---

## 📊 Analysis Notebooks

After running the pipeline scripts, use these notebooks for analysis:

### Main Analysis
- **`notebooks/probe_ssl_embeddings.ipynb`**
  - Linear probing on SSL embeddings (internal validation)
  - Single-task and multi-task probes
  - Uses MassSpecGym test set from Murcko histogram splits

- **`notebooks/external_validation_massbank.ipynb`**
  - External validation on MassBank EU
  - Compares internal R² vs external R²
  - Measures generalization gap

### Exploratory (Archive)
- **`notebooks/exploratory/dataset_creation.ipynb`**
  - Original dataset exploration (not used in current pipeline)
  - Kept for reference only

---

## 🔧 Utility Scripts

- **`scaffold_splits.py`**: Alternative splitting method (GroupShuffleSplit)
- **`simple_probing.py`**: Standalone probing script
- **`add_embeddings_to_tsv.py`**: Convert parquet to TSV format
- **`convert_parquet_to_hdf5.py`**: Convert parquet to HDF5 format

---

## 📁 Expected Data Structure

```
dreams-thesis-wa/
├── data/
│   ├── raw/
│   │   └── MassSpecGym.tsv
│   ├── processed/
│   │   └── massspecgym_complete/
│   │       └── ssl_embs/
│   │           ├── MassSpecGym_with_SSL_embeddings.parquet
│   │           └── MassSpecGym_with_SSL_embeddings_murcko_hist_splits.parquet
│   └── external/
│       └── massbank_eu/
│           ├── massbank_eu_external_test_no_embeddings.parquet
│           └── massbank_eu_external_test.parquet
├── models/
│   └── ssl_model.ckpt
└── results/
    ├── probing_results_ssl.pkl
    └── external_validation_massbank.pkl
```

---

## 🚀 Quick Start

To run the full pipeline from scratch:

```bash
# Step 1: Generate SSL embeddings
python src/generate_ssl_embeddings.py

# Step 2: Add RDKit descriptors
python src/add_rdkit_descriptors.py

# Step 3: Create Murcko histogram splits
python src/murcko_histogram_splits.py

# Step 4 (Optional): Prepare external validation set
python src/prepare_massbank_external_test.py
python src/compute_massbank_embeddings.py

# Step 5: Run analysis notebooks
# - Open notebooks/probe_ssl_embeddings.ipynb
# - Open notebooks/external_validation_massbank.ipynb
```

---

## 📝 Notes

- **Murcko histogram splits are more rigorous** than simple scaffold splits
  - They prevent structurally *similar* molecules from leaking across folds
  - This results in lower R² scores (0.1-0.4 range), which is more realistic and publishable
  
- **External validation** (MassBank EU) has **zero overlap** with MassSpecGym
  - Deduplicated by InChIKey
  - Different MS/MS database, instruments, and acquisition parameters
  - True out-of-distribution test

- **All scripts use `.parquet` format** for efficiency
  - Faster I/O than TSV/CSV
  - Preserves data types (especially for embeddings as arrays)

---

## ❓ Questions?

- Check `EXTERNAL_VALIDATION_GUIDE.md` for detailed external validation workflow
- Check `MEMORY_OPTIMIZATION_GUIDE.md` for training tips
- See `notebooks/exploratory/` for dataset exploration examples
