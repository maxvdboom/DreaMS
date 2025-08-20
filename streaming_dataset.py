#!/usr/bin/env python3
"""
Streaming dataset modification to reduce memory usage.
This modifies the MSData class to avoid loading everything into memory.
"""

import h5py
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class StreamingMSDataset(Dataset):
    """
    Memory-efficient streaming dataset that doesn't load all data into memory.
    """
    
    def __init__(self, hdf5_pth, spec_preproc, label_key, dformat='A', cache_size=1000):
        self.hdf5_pth = Path(hdf5_pth)
        self.spec_preproc = spec_preproc
        self.label_key = label_key
        self.dformat = dformat
        self.cache_size = cache_size
        self.cache = {}
        
        # Open file to get metadata
        with h5py.File(hdf5_pth, 'r') as f:
            self.num_spectra = f['spectrum'].shape[0]
            self.available_keys = list(f.keys())
            
        # Validate required keys exist
        required_keys = ['spectrum', 'precursor_mz', label_key]
        for key in required_keys:
            if key not in self.available_keys:
                raise ValueError(f"Required key '{key}' not found in HDF5 file")
    
    def __len__(self):
        return self.num_spectra
    
    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
        
        # Load from disk
        with h5py.File(self.hdf5_pth, 'r') as f:
            spectrum = f['spectrum'][idx]
            prec_mz = f['precursor_mz'][idx]
            label = f[self.label_key][idx]
            charge = f.get('charge', np.array([1]))[idx] if 'charge' in f else 1
        
        # Preprocess spectrum
        processed_spec = self.spec_preproc(
            spec=spectrum,
            prec_mz=prec_mz,
            high_form=None,
            augment=True
        )
        
        sample = {
            'spec': processed_spec,
            'charge': charge,
            'label': label
        }
        
        # Cache if we have space
        if len(self.cache) < self.cache_size:
            self.cache[idx] = sample
        
        return sample


def create_streaming_dataset(hdf5_pth, spec_preproc, label, dformat='A'):
    """
    Create a streaming dataset that doesn't load everything into memory.
    """
    return StreamingMSDataset(hdf5_pth, spec_preproc, label, dformat)


# Usage in training script:
# Replace the MSData.to_torch_dataset call with:
# dataset = create_streaming_dataset(args.dataset_pth, spec_preproc, args.train_objective, args.dformat)
