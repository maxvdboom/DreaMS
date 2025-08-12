#!/usr/bin/env python3
"""
test_dreams_hdf5.py

Test script to validate that the converted HDF5 file works with DreaMS.
"""

import h5py
import numpy as np
import sys
from pathlib import Path

# Add DreaMS to path
dreams_path = Path(__file__).parent.parent  # Go up to DreaMS root
sys.path.append(str(dreams_path))

try:
    from dreams.utils.data import MSData
    from dreams.definitions import SPECTRUM, PRECURSOR_MZ, CHARGE, ADDUCT, SMILES
    print("Successfully imported DreaMS modules")
except ImportError as e:
    print(f"Failed to import DreaMS modules: {e}")
    print(f"Tried to import from: {dreams_path}")
    print("Make sure you're running this from the DreaMS/scripts directory")
    sys.exit(1)

def test_hdf5_file(hdf5_path):
    """Test that the HDF5 file is compatible with DreaMS."""
    print(f"Testing HDF5 file: {hdf5_path}")
    
    if not Path(hdf5_path).exists():
        print(f"File not found: {hdf5_path}")
        return False
    
    try:
        # Test basic HDF5 structure
        with h5py.File(hdf5_path, 'r') as f:
            print(f"HDF5 file contains columns: {list(f.keys())}")
            print(f"Number of spectra: {f.attrs.get('num_spectra', 'unknown')}")
            
            # Check required columns
            required_cols = [SPECTRUM, PRECURSOR_MZ, CHARGE, ADDUCT, SMILES]
            missing_cols = [col for col in required_cols if col not in f.keys()]
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                return False
            
            # Check shapes
            num_spectra = f[SPECTRUM].shape[0]
            print(f"Spectrum shape: {f[SPECTRUM].shape}")
            print(f"Precursor MZ shape: {f[PRECURSOR_MZ].shape}")
            print(f"Charge shape: {f[CHARGE].shape}")
            print(f"Adduct shape: {f[ADDUCT].shape}")
            print(f"SMILES shape: {f[SMILES].shape}")
            
            # Check if Morgan fingerprints are present
            if 'fp_morgan_4096' in f.keys():
                print(f"Morgan 4096 fingerprints shape: {f['fp_morgan_4096'].shape}")
            else:
                print("Warning: Morgan 4096 fingerprints not found")
        
        # Test MSData loading
        print("\nTesting MSData loading...")
        msdata = MSData(hdf5_path, in_mem=False)
        print(f"MSData loaded successfully with {msdata.num_spectra} spectra")
        print(f"Available columns: {msdata.columns()}")
        
        # Test torch dataset conversion
        print("\nTesting torch dataset conversion...")
        try:
            dataset = msdata.to_torch_dataset(
                label='fp_morgan_4096',
                dformat='default'
            )
            print(f"Torch dataset created successfully with {len(dataset)} samples")
            
            # Test a sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"Sample keys: {sample.keys()}")
                print("Conversion test passed!")
                return True
            
        except Exception as e:
            print(f"Error creating torch dataset: {e}")
            return False
            
    except Exception as e:
        print(f"Error testing HDF5 file: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 test_dreams_hdf5.py <hdf5_file>")
        sys.exit(1)
    
    hdf5_path = sys.argv[1]
    success = test_hdf5_file(hdf5_path)
    
    if success:
        print("\n✅ HDF5 file is compatible with DreaMS!")
    else:
        print("\n❌ HDF5 file has compatibility issues")
        sys.exit(1)

if __name__ == "__main__":
    main()
