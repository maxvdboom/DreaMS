# Quick Start: Simple Probing Setup ✅

## What We Built (Lightweight Version)

A **simple, focused** probing framework that works with **final embeddings only**:

```
✅ notebooks/axis1_probing_simple.ipynb  - Main workflow notebook
✅ src/simple_probing.py                 - Probing utilities (LinearProbe, MLPProbe, kNN, UMAP)
✅ README.md                             - Overview and instructions
✅ PROBING_APPROACH.md                   - Why this approach
✅ requirements.txt                      - Dependencies
```

## Your Workflow

### 1. Get Final Embeddings (YOUR STEP)

You need to extract final layer embeddings from DreaMS model:

```python
# Example: Extract embeddings
from your_dreams_model import DreamsModel
import torch

model = DreamsModel.from_pretrained('path/to/checkpoint')
model.eval()

# Get final embeddings for all spectra
final_embeddings = []
with torch.no_grad():
    for batch in dataloader:
        outputs = model(batch)
        # Get final layer output - adjust based on your model
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token or
        # embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        final_embeddings.append(embeddings.cpu().numpy())

final_embeddings = np.vstack(final_embeddings)
# Save for reuse
np.save('../data/embeddings/final_embeddings.npy', final_embeddings)
```

**Expected shape**: `(n_spectra, embedding_dim)` e.g., `(231104, 768)`

### 2. Run the Notebook

```bash
jupyter notebook notebooks/axis1_probing_simple.ipynb
```

The notebook will:
1. ✅ Load your final embeddings
2. ✅ Load targets from enriched dataset (already has MW, LogP, TPSA, functional groups)
3. ✅ Train linear probes
4. ✅ Train MLP probes  
5. ✅ Compare performance
6. ✅ Validate with kNN retrieval
7. ✅ Visualize with UMAP

### 3. Interpret Results

You'll answer:
- **What's encoded?** Which properties are predictable from embeddings?
- **How well?** AUROC for binary, R² for regression
- **Linear enough?** Compare linear vs MLP probe performance
- **Validated?** Do kNN and UMAP confirm findings?

## Example Output

```
aromatic (linear probe):
  AUROC: 0.873
  AP: 0.891
  Accuracy: 0.812

mol_weight (linear probe):
  R²: 0.654
  RMSE: 0.421

kNN Validation:
  Aromatic - Precision@10: 0.781
```

## What's Different From Before?

**Simpler:**
- ❌ No complex layer-by-layer extraction
- ❌ No hook-based extractors
- ❌ No multi-layer management
- ✅ Just final embeddings
- ✅ Quick to test
- ✅ Easy to debug

**Same goals, simpler path:**
1. Test probing on final embeddings first
2. Validate the approach works
3. **Later** → Expand to per-layer if needed

## Next Steps (After This Works)

Once you have results from final embeddings:

### Option A: Satisfied with final embeddings
- ✅ You know what's encoded
- ✅ Move to Axis 2 (retrieval) or Axis 3 (generation)

### Option B: Want layer-by-layer analysis
- Extract embeddings from each layer
- Run same probing pipeline per layer
- Plot layer-wise curves
- See WHERE information emerges

## Files to Edit

**Critical:**
- `notebooks/axis1_probing_simple.ipynb` → Cell 4: Load your embeddings

**Optional (already working):**
- `src/simple_probing.py` → Probing logic (shouldn't need changes)

## Tips

1. **Test small first**: Use subset of data (1000 samples) to verify everything works
2. **Check alignment**: Ensure embeddings and dataset are in same order
3. **Save embeddings**: Don't re-extract every time
4. **Start with 1-2 targets**: Test aromatic + mol_weight first

## Success Criteria

You're successful when you can say:
- ✅ "Aromatic groups are detectable with AUROC = X"
- ✅ "Molecular weight is encoded with R² = Y"
- ✅ "MLP probes [do/don't] outperform linear"
- ✅ "kNN validation confirms Z"

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/axis1_probing_simple.ipynb
```

## Questions?

Check:
- `README.md` - Overview
- `PROBING_APPROACH.md` - Why this approach
- `src/simple_probing.py` - Implementation details

---

**TL;DR**: Get final embeddings → Run notebook → See what's encoded → Done! 🎯
