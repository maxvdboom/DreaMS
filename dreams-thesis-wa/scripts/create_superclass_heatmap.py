"""Generate heatmap visualization of superclass distribution for thesis.

Dataset structure (mutually exclusive splits):
- Train: development.hdf5 train fold (21,471 molecules)
- Validation: development.hdf5 val fold (6,147 molecules)  
- Holdout: holdout.parquet (3,984 molecules)
- Total: 31,602 unique molecules (no overlap)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import h5py
from matplotlib.colors import LogNorm
from pathlib import Path

# Paths
DATA_DIR = Path('/Users/wouterachterberg/coding/DreaMS/dreams-thesis-wa/data/processed/MassSpecGym_splits')
FIG_DIR = Path('/Users/wouterachterberg/coding/DreaMS/dreams-thesis-wa/figures')
FIG_DIR.mkdir(exist_ok=True)

# Load superclass mapping
smiles_superclass = pd.read_csv(DATA_DIR / 'smiles_with_superclass.csv')

# Load the three mutually exclusive splits
# 1. Train: from development.hdf5 train fold (full minus holdout)
with h5py.File(DATA_DIR / 'development.hdf5', 'r') as hf:
    dev_smiles = [s.decode() if isinstance(s, bytes) else s for s in hf['smiles'][:]]
    dev_folds = [f.decode() if isinstance(f, bytes) else f for f in hf['fold'][:]]
dev_df = pd.DataFrame({'smiles': dev_smiles, 'fold': dev_folds})
train_smiles = set(dev_df[dev_df['fold'] == 'train']['smiles'].unique())

# 2. Validation: from development.hdf5 val fold (= probing_test.parquet)
val_smiles = set(dev_df[dev_df['fold'] == 'val']['smiles'].unique())

# 3. Holdout: from holdout.parquet
holdout = pq.read_table(DATA_DIR / 'holdout.parquet').to_pandas()
holdout_smiles = set(holdout['smiles'].unique())

# Verify no overlap
assert len(train_smiles & val_smiles) == 0, "Train and Val overlap!"
assert len(train_smiles & holdout_smiles) == 0, "Train and Holdout overlap!"
assert len(val_smiles & holdout_smiles) == 0, "Val and Holdout overlap!"
total_unique = len(train_smiles | val_smiles | holdout_smiles)
print(f"Train: {len(train_smiles):,}, Val: {len(val_smiles):,}, Holdout: {len(holdout_smiles):,}")
print(f"Total unique (no overlap): {total_unique:,}")

# Get superclass counts for each split
def get_superclass_counts(smiles_set):
    df = pd.DataFrame({'smiles': list(smiles_set)})
    merged = df.merge(smiles_superclass, on='smiles', how='left')
    return merged['superclass'].value_counts()

train_counts = get_superclass_counts(train_smiles)
val_counts = get_superclass_counts(val_smiles)
holdout_counts = get_superclass_counts(holdout_smiles)

# Combine into DataFrame
all_superclasses = set(train_counts.index) | set(val_counts.index) | set(holdout_counts.index)
df = pd.DataFrame(index=sorted(all_superclasses))
df['Train'] = train_counts
df['Validation'] = val_counts
df['Holdout'] = holdout_counts
df = df.fillna(0).astype(int)
df['Total'] = df.sum(axis=1)
df = df.sort_values('Total', ascending=False)

# Save updated CSV with correct splits
df['Train %'] = (df['Train'] / df['Train'].sum() * 100).round(2)
df['Val %'] = (df['Validation'] / df['Validation'].sum() * 100).round(2)
df['Holdout %'] = (df['Holdout'] / df['Holdout'].sum() * 100).round(2)
df['Total %'] = (df['Total'] / df['Total'].sum() * 100).round(2)
df.to_csv(DATA_DIR / 'superclass_distribution_full.csv')
print(f"Saved: {DATA_DIR / 'superclass_distribution_full.csv'}")

# Take top 15 superclasses for readability
top_n = 15
top = df.head(top_n).copy()

# Create a matrix for the heatmap (using raw counts)
heatmap_data = top[['Train', 'Validation', 'Holdout']].copy()
n_train = df['Train'].sum()
n_val = df['Validation'].sum()
n_holdout = df['Holdout'].sum()
heatmap_data.columns = [f'Train\n(n={n_train:,})', f'Validation\n(n={n_val:,})', f'Holdout\n(n={n_holdout:,})']

# Create figure
fig, ax = plt.subplots(figsize=(9, 9))

# Create heatmap with log scale for better color distribution
# Handle zeros by replacing with 0.5 for log scale
heatmap_values = heatmap_data.values.astype(float)
heatmap_values[heatmap_values == 0] = 0.5  # Small value for log scale

im = ax.imshow(heatmap_values, cmap='Blues', aspect='auto', 
               norm=LogNorm(vmin=0.5, vmax=heatmap_data.values.max()))

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
cbar.set_label('Number of molecules (log scale)', fontsize=10)

# Set ticks
ax.set_xticks(range(3))
ax.set_xticklabels(heatmap_data.columns, fontsize=11, fontweight='bold')
ax.set_yticks(range(len(heatmap_data)))
ax.set_yticklabels(heatmap_data.index, fontsize=10)

# Add text annotations with counts
for i in range(len(heatmap_data)):
    for j in range(3):
        val = int(heatmap_data.iloc[i, j])
        # Use white text for dark cells, black for light
        color = 'white' if val > 300 else 'black'
        ax.text(j, i, f'{val:,}', ha='center', va='center', 
                fontsize=9, color=color, fontweight='bold')

# Add note about Other classes
other_train = df.iloc[top_n:]['Train'].sum()
other_val = df.iloc[top_n:]['Validation'].sum()
other_holdout = df.iloc[top_n:]['Holdout'].sum()
fig.text(0.5, 0.02, 
         f'Note: {len(df)-top_n} additional classes not shown (Train: {other_train:,}, Val: {other_val:,}, Holdout: {other_holdout:,})',
         ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)

# Save
plt.savefig(FIG_DIR / 'superclass_heatmap.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig(FIG_DIR / 'superclass_heatmap.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f'Saved: {FIG_DIR / "superclass_heatmap.png"}')
print(f'Saved: {FIG_DIR / "superclass_heatmap.pdf"}')
