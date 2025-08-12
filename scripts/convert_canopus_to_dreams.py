#!/usr/bin/env python3
"""
convert_canopus_to_dreams.py

Convert CANOPUS benchmark dataset from MIST format to DreaMS HDF5 format.
This script converts the MIST-format CANOPUS dataset to the HDF5 format expected by DreaMS.
"""

import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle
import sys
import os

# Add MIST to path for data loading utilities
sys.path.append(str(Path(__file__).parent.parent / "mist" / "src"))

from mist.data.datasets import get_paired_spectra
from mist.data.featurizers import FingerprintFeaturizer
from mist import utils

def load_canopus_data(canopus_dir):
    """Load CANOPUS data using MIST utilities."""
    canopus_dir = Path(canopus_dir)
    
    # Find the labels file
    labels_file = canopus_dir / "labels.tsv"
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    # Find spectra directory - try multiple possible locations
    spec_dirs_to_try = [
        canopus_dir / "spec",
        canopus_dir / "spectra", 
        canopus_dir,
        canopus_dir / "ms"
    ]
    
    spec_dir = None
    for potential_dir in spec_dirs_to_try:
        if potential_dir.exists():
            # Check if it contains .ms files
            ms_files = list(potential_dir.glob("*.ms"))
            if ms_files:
                spec_dir = potential_dir
                print(f"Found {len(ms_files)} .ms files in {spec_dir}")
                break
    
    if spec_dir is None:
        # List all files to help debug
        print(f"Directory contents of {canopus_dir}:")
        for item in canopus_dir.iterdir():
            print(f"  {item.name} ({'dir' if item.is_dir() else 'file'})")
        raise FileNotFoundError(f"No .ms files found in any expected directory under {canopus_dir}")
    
    print(f"Loading CANOPUS data from {canopus_dir}")
    print(f"Labels file: {labels_file}")
    print(f"Spectra directory: {spec_dir}")
    
    # Load paired spectra using MIST utilities
    spectra_list, mol_list = get_paired_spectra(
        labels_file=str(labels_file),
        spec_folder=str(spec_dir),
        prog_bars=True
    )
    
    print(f"Loaded {len(spectra_list)} spectra and {len(mol_list)} molecules")
    
    if len(spectra_list) == 0:
        raise ValueError("No spectra were loaded. Check that the dataset format is correct.")
    
    return spectra_list, mol_list

def compute_morgan_fingerprints(smiles_list, fp_names=["morgan4096"]):
    """Compute Morgan fingerprints for SMILES."""
    print("Computing Morgan fingerprints...")
    
    featurizer = FingerprintFeaturizer(fp_names=fp_names)
    fingerprints = []
    
    for smiles in tqdm(smiles_list, desc="Computing fingerprints"):
        if smiles and smiles != "nan":
            try:
                fp = featurizer.featurize_smiles(smiles)
                fingerprints.append(fp)
            except Exception as e:
                print(f"Error computing fingerprint for {smiles}: {e}")
                fingerprints.append(np.zeros(4096, dtype=np.uint8))
        else:
            fingerprints.append(np.zeros(4096, dtype=np.uint8))
    
    return np.array(fingerprints)

def convert_spectrum_format(spectrum_data):
    """Convert MIST spectrum format to DreaMS format."""
    # MIST stores spectra as lists of [mz, intensity] pairs
    # DreaMS expects shape (num_spectra, 2, num_peaks) with padding
    
    if len(spectrum_data) == 0:
        print("Warning: No spectra to convert")
        return np.array([]).reshape(0, 2, 512)  # Return empty array with correct shape
    
    max_peaks = 512  # Standard for DreaMS
    converted_spectra = []
    
    for spec in tqdm(spectrum_data, desc="Converting spectra"):
        try:
            if hasattr(spec, 'peaks') and spec.peaks is not None:
                peaks = spec.peaks
                if len(peaks) > 0:
                    # Extract m/z and intensity arrays
                    mz_array = np.array([p[0] for p in peaks], dtype=np.float32)
                    intensity_array = np.array([p[1] for p in peaks], dtype=np.float32)
                    
                    # Take top max_peaks by intensity
                    if len(peaks) > max_peaks:
                        top_indices = np.argpartition(intensity_array, -max_peaks)[-max_peaks:]
                        mz_array = mz_array[top_indices]
                        intensity_array = intensity_array[top_indices]
                    
                    # Sort by m/z
                    sort_indices = np.argsort(mz_array)
                    mz_array = mz_array[sort_indices]
                    intensity_array = intensity_array[sort_indices]
                    
                    # Pad to max_peaks
                    if len(mz_array) < max_peaks:
                        pad_length = max_peaks - len(mz_array)
                        mz_array = np.pad(mz_array, (0, pad_length), mode='constant', constant_values=0)
                        intensity_array = np.pad(intensity_array, (0, pad_length), mode='constant', constant_values=0)
                    
                    # Stack as (2, max_peaks) - first row m/z, second row intensity
                    spectrum_matrix = np.stack([mz_array, intensity_array], axis=0)
                    converted_spectra.append(spectrum_matrix)
                else:
                    # Empty spectrum
                    converted_spectra.append(np.zeros((2, max_peaks), dtype=np.float32))
            else:
                # No spectrum data
                converted_spectra.append(np.zeros((2, max_peaks), dtype=np.float32))
        except Exception as e:
            print(f"Error converting spectrum: {e}")
            # Add empty spectrum as fallback
            converted_spectra.append(np.zeros((2, max_peaks), dtype=np.float32))
    
    if len(converted_spectra) == 0:
        print("Warning: No valid spectra after conversion")
        return np.array([]).reshape(0, 2, max_peaks)
    
    return np.array(converted_spectra)

def create_dreams_hdf5(spectra_list, mol_list, output_path, include_fingerprints=True):
    """Create DreaMS-compatible HDF5 file."""
    print(f"Creating DreaMS HDF5 file: {output_path}")
    
    # Extract data from MIST objects
    precursor_mzs = []
    charges = []
    adducts = []
    smiles_list = []
    
    for i, (spectrum, mol) in enumerate(zip(spectra_list, mol_list)):
        # Extract precursor m/z
        if hasattr(spectrum, 'precursor_mz') and spectrum.precursor_mz is not None:
            precursor_mzs.append(float(spectrum.precursor_mz))
        else:
            precursor_mzs.append(0.0)
        
        # Extract charge (default to 1 if not available)
        if hasattr(spectrum, 'charge') and spectrum.charge is not None:
            charges.append(int(spectrum.charge))
        else:
            charges.append(1)
        
        # Extract adduct (default to [M+H]+ if not available)
        if hasattr(spectrum, 'adduct') and spectrum.adduct is not None:
            adducts.append(str(spectrum.adduct))
        else:
            adducts.append("[M+H]+")
        
        # Extract SMILES
        if hasattr(mol, 'smiles') and mol.smiles is not None:
            smiles_list.append(str(mol.smiles))
        else:
            smiles_list.append("")
    
    # Convert spectra to DreaMS format
    spectrum_data = convert_spectrum_format(spectra_list)
    
    # Validate spectrum data
    if len(spectrum_data) == 0:
        raise ValueError("No valid spectra found after conversion")
    
    print(f"Converted {len(spectrum_data)} spectra to DreaMS format")
    print(f"Spectrum shape: {spectrum_data.shape}")
    
    # Compute Morgan fingerprints if requested
    fingerprints = None
    if include_fingerprints:
        fingerprints = compute_morgan_fingerprints(smiles_list)
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Core DreaMS columns
        f.create_dataset('spectrum', data=spectrum_data, compression='gzip')
        f.create_dataset('precursor_mz', data=np.array(precursor_mzs, dtype=np.float32), compression='gzip')
        f.create_dataset('charge', data=np.array(charges, dtype=np.int32), compression='gzip')
        f.create_dataset('adduct', data=[s.encode('utf-8') for s in adducts], compression='gzip')
        f.create_dataset('smiles', data=[s.encode('utf-8') for s in smiles_list], compression='gzip')
        
        # Add Morgan fingerprints for fine-tuning
        if fingerprints is not None:
            f.create_dataset('fp_morgan_4096', data=fingerprints.astype(np.uint8), compression='gzip')
        
        # Add metadata
        f.attrs['dataset'] = 'CANOPUS'
        f.attrs['num_spectra'] = len(spectra_list)
        f.attrs['max_peaks'] = spectrum_data.shape[2] if len(spectrum_data.shape) >= 3 else 512
        f.attrs['converted_from'] = 'MIST'
        
        print(f"Successfully created HDF5 file with {len(spectra_list)} spectra")
        print(f"Columns: {list(f.keys())}")

def main():
    parser = argparse.ArgumentParser(description="Convert CANOPUS MIST dataset to DreaMS HDF5 format")
    parser.add_argument("canopus_dir", help="Path to CANOPUS dataset directory")
    parser.add_argument("output_path", help="Output HDF5 file path")
    parser.add_argument("--no-fingerprints", action="store_true", help="Don't compute Morgan fingerprints")
    
    args = parser.parse_args()
    
    # Validate inputs
    canopus_dir = Path(args.canopus_dir)
    if not canopus_dir.exists():
        raise FileNotFoundError(f"CANOPUS directory not found: {canopus_dir}")
    
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load CANOPUS data
        spectra_list, mol_list = load_canopus_data(canopus_dir)
        
        # Create DreaMS HDF5 file
        create_dreams_hdf5(
            spectra_list, 
            mol_list, 
            output_path, 
            include_fingerprints=not args.no_fingerprints
        )
        
        print(f"\nConversion completed successfully!")
        print(f"DreaMS HDF5 file created: {output_path}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
