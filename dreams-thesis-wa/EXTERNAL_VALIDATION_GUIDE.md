# MassBank EU External Validation Setup

This guide walks you through setting up an external test set from **MassBank EU** to validate your DreaMS SSL embeddings on completely unseen data.

## Why MassBank EU?

- ✅ **Clean OOD check**: Different repository than MassSpecGym (trained on GNPS/MassIVE)
- ✅ **InChIKey/SMILES available**: Can compute RDKit descriptors directly
- ✅ **Moderate size**: ~200-500 compounds sufficient for stable linear/MLP probing
- ✅ **Similar instruments**: Can filter for Orbitrap/QTOF (comparable to MassSpecGym)
- ✅ **Public & curated**: High-quality MS/MS data

## Workflow Overview

```
1. Download MassBank EU data
   └─> fetch_massbank_api.py (API approach)
   └─> OR manual download from GitHub

2. Filter & deduplicate
   └─> prepare_massbank_external_test.py
   └─> Removes overlap with MassSpecGym training data

3. Compute SSL embeddings
   └─> compute_massbank_embeddings.py (uses frozen DreaMS encoder)

4. Run external validation
   └─> evaluate_external_test.ipynb
   └─> Compare internal test R² vs external R²
```

---

## Option 1: API Fetch (Simpler, Slower)

### Step 1: Fetch records via MassBank API

```bash
cd dreams-thesis-wa
../.venv/bin/python src/fetch_massbank_api.py --max-compounds 500
```

This will:
- Query MassBank EU API for positive ESI MS/MS spectra
- Download metadata + spectra for ~500 compounds
- Save to `data/external/massbank_eu/massbank_raw_records.json`
- Takes ~10-15 minutes (API rate limiting)

---

## Option 2: Bulk Download (Faster, Larger)

### Step 1: Download MassBank-data repository

```bash
cd dreams-thesis-wa/data/external/massbank_eu
wget https://github.com/MassBank/MassBank-data/archive/refs/heads/main.zip
unzip main.zip
```

### Step 2: Run preparation script

```bash
cd ../../..  # Back to dreams-thesis-wa
../.venv/bin/python src/prepare_massbank_external_test.py
```

This will:
- Parse MassBank `.txt` records
- Filter for: positive ESI, Orbitrap/QTOF, mid CE (20-40 eV)
- **Deduplicate by InChIKey** against your MassSpecGym training data
- Compute 10 RDKit descriptors
- Save to `data/external/massbank_eu/massbank_eu_external_test_no_embeddings.parquet`

**Expected output**: ~200-500 unique compounds

---

## Step 3: Compute SSL Embeddings

You'll need to compute DreaMS SSL embeddings for the MassBank spectra using your **frozen encoder** (same as used for MassSpecGym).

### Create embedding generation script

I'll create `compute_massbank_embeddings.py` that:
- Loads your `ssl_model.ckpt`
- Processes MassBank spectra through the frozen encoder
- Saves embeddings to final parquet file

```bash
../.venv/bin/python src/compute_massbank_embeddings.py
```

---

## Step 4: External Validation (Probing)

Once embeddings are computed, run the external validation notebook:

```python
# In evaluate_external_test.ipynb

# Load trained probes from MassSpecGym (frozen weights)
with open('../results/probing_results_ssl.pkl', 'rb') as f:
    trained_models = pickle.load(f)

# Load MassBank external test set
massbank_df = pd.read_parquet('../data/external/massbank_eu/massbank_eu_external_test.parquet')

# Evaluate each probe on external data (NO RETRAINING!)
for descriptor in descriptors:
    X_test_external = massbank_df['ssl_embedding']
    y_test_external = massbank_df[descriptor]
    
    # Use FIXED probe trained on MassSpecGym
    model = trained_models[descriptor]['best_model']
    
    # Evaluate
    external_r2 = evaluate_probe(model, X_test_external, y_test_external)
    
    print(f"{descriptor}: Internal R²={internal_r2:.3f}, External R²={external_r2:.3f}")
```

---

## Expected Results

### Strong Generalization (Good SSL Embeddings)
```
Descriptor        Internal Test R²    External Test R²
─────────────────────────────────────────────────────
alogp                    0.847              0.812     ✅ -0.035
hba                      0.623              0.591     ✅ -0.032
tpsa                     0.715              0.683     ✅ -0.032
aromatic_rings           0.782              0.741     ✅ -0.041
```
**→ Small drop (< 0.05) = good generalization**

### Weak Generalization (Overfitting/Distribution Shift)
```
Descriptor        Internal Test R²    External Test R²
─────────────────────────────────────────────────────
alogp                    0.847              0.612     ⚠️  -0.235
hba                      0.623              0.401     ⚠️  -0.222
```
**→ Large drop (> 0.15) = embeddings may overfit to MassSpecGym**

---

## Key Points for Thesis

### Validation Strategy

In your thesis, describe it as:

> **Internal Validation**: We evaluated probes on a held-out test set from MassSpecGym (19% of data, n=45,185 spectra), ensuring no Murcko scaffold overlap with training data.
>
> **External Validation**: To assess generalization to unseen instruments and databases, we validated the same probes (with frozen weights) on an external test set from MassBank EU (n=~300-500 unique compounds), filtered to match MassSpecGym instrument conditions and deduplicated by InChIKey.

### Expected Findings

1. **Internal test R²**: Reports probe performance on same-distribution data
2. **External validation R²**: Tests if SSL embeddings encode generalizable molecular features
3. **R² drop analysis**: Quantifies distribution shift / domain gap

### Terminology

- ✅ "External validation set" (most common)
- ✅ "External test set" (also correct)
- ✅ "Out-of-distribution (OOD) evaluation"
- ❌ "External training set" (never retrain!)

---

## Troubleshooting

### API fetch fails
→ Use bulk download (Option 2)

### Not enough compounds after deduplication
→ Relax filters (accept QTOF only, wider CE range)
→ Or use multiple MassBank repositories (JP, NA)

### Embeddings don't generate
→ Check that `ssl_model.ckpt` path is correct
→ Verify DreaMS model loads properly

### Large R² drop (> 0.20)
→ Expected! Different instruments, acquisition methods
→ Discuss as "domain adaptation" challenge
→ Consider if MassSpecGym is instrument-specific

---

## Next Steps

1. **Run Step 2**: Parse and deduplicate MassBank data
2. **Check sample size**: Should have ~200-500 unique compounds
3. **I'll create**: `compute_massbank_embeddings.py` for Step 3
4. **You run**: External validation notebook for Step 4

Would you like me to:
- Create the embedding computation script?
- Set up the external validation notebook?
- Help run the MassBank download?
