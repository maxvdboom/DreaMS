#!/usr/bin/env python3
"""
Standalone script for probing ALL RDKit descriptors on GPU cluster.
This script trains both Linear and MLP probes for ~200 descriptors.

Usage:
    python probe_all_descriptors_cluster.py --device cuda:0

Outputs:
    - results/all_descriptors_probing_results_linear.pkl
    - results/all_descriptors_probing_results_mlp.pkl
    - results/cluster_run_summary.txt
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
from datetime import datetime
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Configuration
# =============================================================================

# Paths (adjust these for your cluster environment)
DATA_PATH = Path('../data/processed/massspecgym_complete/ssl_embs/MassSpecGym_with_SSL_embeddings_murcko_hist_splits.parquet')
DESCRIPTORS_PATH = Path('../data/processed/massspecgym_complete/all_rdkit_descriptors.parquet')
RESULTS_DIR = Path('../results')

# Model parameters
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 30
DROPOUT = 0.2

# =============================================================================
# Model Definitions
# =============================================================================

class LinearProbe(nn.Module):
    """Simple linear regression probe: 1024→1"""
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


class MLPProbe(nn.Module):
    """3-layer MLP probe with ReLU and Dropout: 1024→256→128→1"""
    def __init__(self, input_dim, output_dim=1, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)


class EmbeddingDataset(Dataset):
    """PyTorch Dataset for embeddings and targets"""
    def __init__(self, embeddings, targets):
        self.embeddings = torch.FloatTensor(embeddings)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]


# =============================================================================
# Training Function
# =============================================================================

def train_probe(X_train, y_train, X_test, y_test, model_type='mlp', 
                epochs=30, lr=0.001, device='cuda:0', verbose=False):
    """
    Train probe and evaluate.
    
    Args:
        X_train: Training embeddings (N, 1024)
        y_train: Training targets (N,)
        X_test: Test embeddings (M, 1024)
        y_test: Test targets (M,)
        model_type: 'linear' or 'mlp'
        epochs: Number of training epochs
        lr: Learning rate
        device: 'cuda:0', 'cuda:1', or 'cpu'
        verbose: Print training progress
    
    Returns:
        dict with r2, mae, model, scaler
    """
    # Standardize targets
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    # Create datasets
    train_dataset = EmbeddingDataset(X_train, y_train_scaled)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create model
    if model_type == 'linear':
        model = LinearProbe(input_dim=X_train.shape[1])
    else:
        model = MLPProbe(input_dim=X_train.shape[1], dropout=DROPOUT)
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for embeddings, targets in train_loader:
            embeddings, targets = embeddings.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        preds_scaled = model(X_test_tensor).cpu().numpy().flatten()
        preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    return {
        'r2': r2,
        'mae': mae,
        'model': model.cpu(),  # Move back to CPU for saving
        'scaler': scaler
    }


# =============================================================================
# Main Probing Function
# =============================================================================

def probe_all_descriptors(device='cuda:0', skip_linear=False, skip_mlp=False):
    """
    Main function to probe all descriptors.
    
    Args:
        device: Device to use ('cuda:0', 'cuda:1', 'cpu')
        skip_linear: Skip linear probing if results exist
        skip_mlp: Skip MLP probing if results exist
    """
    print("="*80)
    print("COMPREHENSIVE RDKIT DESCRIPTOR PROBING")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print("="*80 + "\n")
    
    # Check if GPU is available
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df_descriptors = pd.read_parquet(DESCRIPTORS_PATH)
    print(f"  Dataset: {len(df):,} spectra")
    print(f"  Descriptors: {len(df_descriptors.columns)-1} total\n")
    
    # Filter valid descriptors
    print("Filtering valid descriptors...")
    descriptor_names = [col for col in df_descriptors.columns if col != 'smiles']
    valid_descriptors = []
    
    for desc in descriptor_names:
        values = df_descriptors[desc]
        nan_pct = values.isna().sum() / len(values)
        
        if nan_pct > 0.1:
            continue
        if np.isinf(values.dropna()).any():
            continue
        if values.dropna().std() == 0:
            continue
        
        valid_descriptors.append(desc)
    
    print(f"  Valid descriptors: {len(valid_descriptors)}/{len(descriptor_names)}\n")
    
    # Merge descriptors with dataset
    print("Merging descriptors with dataset...")
    # Drop any overlapping descriptor columns from the main dataset first
    overlapping_cols = [col for col in valid_descriptors if col in df.columns and col != 'smiles']
    if overlapping_cols:
        print(f"  Dropping {len(overlapping_cols)} overlapping columns: {overlapping_cols[:5]}...")
        df = df.drop(columns=overlapping_cols)
    
    df_with_descriptors = df.merge(
        df_descriptors[['smiles'] + valid_descriptors],
        on='smiles',
        how='left'
    )
    
    # Split by fold
    df_train = df_with_descriptors[df_with_descriptors['fold'] == 'train'].copy()
    df_test = df_with_descriptors[df_with_descriptors['fold'] == 'test'].copy()
    print(f"  Train: {len(df_train):,}")
    print(f"  Test:  {len(df_test):,}\n")
    
    # Extract embeddings
    print("Extracting embeddings...")
    X_train = np.vstack(df_train['ssl_embedding'].values)
    X_test = np.vstack(df_test['ssl_embedding'].values)
    print(f"  Train embeddings: {X_train.shape}")
    print(f"  Test embeddings:  {X_test.shape}\n")
    
    # =============================================================================
    # LINEAR PROBE
    # =============================================================================
    
    results_cache_linear = RESULTS_DIR / 'all_descriptors_probing_results_linear.pkl'
    
    if results_cache_linear.exists() and skip_linear:
        print(f"✓ Skipping Linear probing (results exist)\n")
        with open(results_cache_linear, 'rb') as f:
            all_results_linear = pickle.load(f)
    else:
        print("="*80)
        print("LINEAR PROBE")
        print("="*80)
        print(f"Descriptors: {len(valid_descriptors)}")
        print(f"Estimated time: ~30-60 minutes")
        print("="*80 + "\n")
        
        all_results_linear = []
        
        for i, desc in enumerate(tqdm(valid_descriptors, desc="Linear probing"), 1):
            # Get targets (drop NaN)
            train_mask = df_train[desc].notna()
            test_mask = df_test[desc].notna()
            
            if train_mask.sum() < 100 or test_mask.sum() < 100:
                continue
            
            y_train = df_train[train_mask][desc].values
            y_test = df_test[test_mask][desc].values
            X_train_clean = X_train[train_mask]
            X_test_clean = X_test[test_mask]
            
            try:
                result = train_probe(
                    X_train_clean, y_train,
                    X_test_clean, y_test,
                    model_type='linear',
                    epochs=EPOCHS,
                    lr=LEARNING_RATE,
                    device=device,
                    verbose=False
                )
                
                all_results_linear.append({
                    'descriptor': desc,
                    'r2': result['r2'],
                    'mae': result['mae'],
                    'n_train': len(y_train),
                    'n_test': len(y_test),
                    'train_mean': y_train.mean(),
                    'train_std': y_train.std(),
                    'test_mean': y_test.mean(),
                    'test_std': y_test.std()
                })
                
            except Exception as e:
                print(f"\n⚠️  Failed on {desc}: {e}")
                continue
        
        # Save Linear results
        with open(results_cache_linear, 'wb') as f:
            pickle.dump(all_results_linear, f)
        
        print(f"\n✅ Linear probing complete!")
        print(f"   Evaluated: {len(all_results_linear)} descriptors")
        print(f"   Saved to: {results_cache_linear.name}\n")
    
    # =============================================================================
    # MLP PROBE
    # =============================================================================
    
    results_cache_mlp = RESULTS_DIR / 'all_descriptors_probing_results_mlp.pkl'
    
    if results_cache_mlp.exists() and skip_mlp:
        print(f"✓ Skipping MLP probing (results exist)\n")
        with open(results_cache_mlp, 'rb') as f:
            all_results_mlp = pickle.load(f)
    else:
        print("="*80)
        print("MLP PROBE")
        print("="*80)
        print(f"Descriptors: {len(valid_descriptors)}")
        print(f"Estimated time: ~60-120 minutes")
        print("="*80 + "\n")
        
        all_results_mlp = []
        
        for i, desc in enumerate(tqdm(valid_descriptors, desc="MLP probing"), 1):
            # Get targets (drop NaN)
            train_mask = df_train[desc].notna()
            test_mask = df_test[desc].notna()
            
            if train_mask.sum() < 100 or test_mask.sum() < 100:
                continue
            
            y_train = df_train[train_mask][desc].values
            y_test = df_test[test_mask][desc].values
            X_train_clean = X_train[train_mask]
            X_test_clean = X_test[test_mask]
            
            try:
                result = train_probe(
                    X_train_clean, y_train,
                    X_test_clean, y_test,
                    model_type='mlp',
                    epochs=EPOCHS,
                    lr=LEARNING_RATE,
                    device=device,
                    verbose=False
                )
                
                all_results_mlp.append({
                    'descriptor': desc,
                    'r2': result['r2'],
                    'mae': result['mae'],
                    'n_train': len(y_train),
                    'n_test': len(y_test),
                    'train_mean': y_train.mean(),
                    'train_std': y_train.std(),
                    'test_mean': y_test.mean(),
                    'test_std': y_test.std()
                })
                
            except Exception as e:
                print(f"\n⚠️  Failed on {desc}: {e}")
                continue
        
        # Save MLP results
        with open(results_cache_mlp, 'wb') as f:
            pickle.dump(all_results_mlp, f)
        
        print(f"\n✅ MLP probing complete!")
        print(f"   Evaluated: {len(all_results_mlp)} descriptors")
        print(f"   Saved to: {results_cache_mlp.name}\n")
    
    # =============================================================================
    # Summary
    # =============================================================================
    
    print("="*80)
    print("PROBING COMPLETE!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to:")
    print(f"  - {results_cache_linear}")
    print(f"  - {results_cache_mlp}")
    print("\nNext steps:")
    print("  1. Download these .pkl files to your local machine")
    print("  2. Run the analysis cells in the Jupyter notebook")
    print("="*80)
    
    # Save summary
    summary_path = RESULTS_DIR / 'cluster_run_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE RDKIT DESCRIPTOR PROBING - CLUSTER RUN SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Dropout: {DROPOUT}\n\n")
        f.write(f"Dataset size: {len(df):,} spectra\n")
        f.write(f"Train size: {len(df_train):,}\n")
        f.write(f"Test size: {len(df_test):,}\n\n")
        f.write(f"Total descriptors: {len(descriptor_names)}\n")
        f.write(f"Valid descriptors: {len(valid_descriptors)}\n")
        f.write(f"Linear results: {len(all_results_linear)}\n")
        f.write(f"MLP results: {len(all_results_mlp)}\n\n")
        f.write("Output files:\n")
        f.write(f"  - {results_cache_linear.name}\n")
        f.write(f"  - {results_cache_mlp.name}\n")
        f.write("="*80 + "\n")
    
    print(f"\n✅ Summary saved to: {summary_path}\n")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Probe ALL RDKit descriptors with Linear and MLP models"
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda:0',
        help='Device to use (cuda:0, cuda:1, or cpu)'
    )
    parser.add_argument(
        '--skip-linear',
        action='store_true',
        help='Skip linear probing if results exist'
    )
    parser.add_argument(
        '--skip-mlp',
        action='store_true',
        help='Skip MLP probing if results exist'
    )
    
    args = parser.parse_args()
    
    try:
        probe_all_descriptors(
            device=args.device,
            skip_linear=args.skip_linear,
            skip_mlp=args.skip_mlp
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
