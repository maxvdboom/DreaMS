#!/usr/bin/env python3
"""
simple_canopus_converter.py

Simple converter that reads CANOPUS data directly without MIST dependencies.
This is a fallback solution if the MIST-based converter doesn't work.
"""

import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re
import sys

def parse_ms_file(ms_file_path):
    """Parse a single .ms file in NIST format."""
    spectrum_data = {
        'peaks': [],
        'precursor_mz': None,
        'charge': 1,
        'adduct': '[M+H]+',
        'name': None
    }
    
    try:
        with open(ms_file_path, 'r') as f:
            lines = f.readlines()
        
        in_spectrum = False
        for line in lines:
            line = line.strip()
            
            if line.startswith('Name:'):
                spectrum_data['name'] = line.split(':', 1)[1].strip()
            elif line.startswith('MW:') or line.startswith('ExactMass:'):
                try:
                    spectrum_data['precursor_mz'] = float(line.split(':', 1)[1].strip())
                except:
                    pass
            elif line.startswith('PrecursorMZ:'):
                try:
                    spectrum_data['precursor_mz'] = float(line.split(':', 1)[1].strip())
                except:
                    pass
            elif line.startswith('Charge:'):
                try:
                    spectrum_data['charge'] = int(line.split(':', 1)[1].strip())
                except:
                    pass
            elif line.startswith('Adduct:'):
                spectrum_data['adduct'] = line.split(':', 1)[1].strip()
            elif line.startswith('Num Peaks:'):
                in_spectrum = True
                continue
            elif in_spectrum and line:
                # Parse peaks: "mz intensity" format
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        spectrum_data['peaks'].append([mz, intensity])
                    except:
                        pass
        
        return spectrum_data
    except Exception as e:
        print(f"Error parsing {ms_file_path}: {e}")
        return None

def load_canopus_simple(canopus_dir):
    """Load CANOPUS data without MIST dependencies."""
    canopus_dir = Path(canopus_dir)
    
    # Load labels
    labels_file = canopus_dir / "labels.tsv"
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    labels_df = pd.read_csv(labels_file, sep="\t").astype(str)
    print(f"Loaded labels with columns: {list(labels_df.columns)}")
    print(f"Number of entries: {len(labels_df)}")
    
    # Create spec name to metadata mapping
    name_to_metadata = {}
    for _, row in labels_df.iterrows():
        spec_name = row['spec']
        name_to_metadata[spec_name] = {
            'formula': row.get('formula', ''),
            'smiles': row.get('smiles', ''),
            'inchikey': row.get('inchikey', ''),
            'instrument': row.get('instrument', '')
        }
    
    # Find .ms files
    ms_files = list(canopus_dir.rglob("*.ms"))
    print(f"Found {len(ms_files)} .ms files")
    
    if len(ms_files) == 0:
        raise FileNotFoundError(f"No .ms files found in {canopus_dir}")
    
    # Parse spectra
    spectra_data = []
    mol_data = []
    
    for ms_file in tqdm(ms_files, desc="Parsing .ms files"):
        spec_name = ms_file.stem
        spectrum = parse_ms_file(ms_file)
        
        if spectrum is not None and spec_name in name_to_metadata:
            metadata = name_to_metadata[spec_name]
            
            # Create spectrum object
            spectra_data.append({
                'name': spec_name,
                'peaks': spectrum['peaks'],
                'precursor_mz': spectrum['precursor_mz'],
                'charge': spectrum['charge'],
                'adduct': spectrum['adduct']
            })
            
            # Create molecule object
            mol_data.append({
                'smiles': metadata['smiles'],
                'formula': metadata['formula'],
                'inchikey': metadata['inchikey']
            })
    
    print(f"Successfully parsed {len(spectra_data)} spectra")
    return spectra_data, mol_data

def compute_morgan_simple(smiles_list):
    """Compute Morgan fingerprints using RDKit directly."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        print("Using RDKit for Morgan fingerprints")
    except ImportError:
        print("RDKit not available, creating dummy fingerprints")
        return np.zeros((len(smiles_list), 4096), dtype=np.uint8)
    
    fingerprints = []
    for smiles in tqdm(smiles_list, desc="Computing Morgan fingerprints"):
        if smiles and smiles != "nan" and smiles != "":
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096)
                    fp_array = np.array(fp, dtype=np.uint8)
                    fingerprints.append(fp_array)
                else:
                    fingerprints.append(np.zeros(4096, dtype=np.uint8))
            except Exception as e:
                print(f"Error computing fingerprint for {smiles}: {e}")
                fingerprints.append(np.zeros(4096, dtype=np.uint8))
        else:
            fingerprints.append(np.zeros(4096, dtype=np.uint8))
    
    return np.array(fingerprints)

def create_dreams_hdf5_simple(spectra_data, mol_data, output_path):
    """Create DreaMS HDF5 file from simple data structures."""
    print(f"Creating DreaMS HDF5 file: {output_path}")
    
    max_peaks = 512
    num_spectra = len(spectra_data)
    
    # Prepare arrays
    spectrum_array = np.zeros((num_spectra, 2, max_peaks), dtype=np.float32)
    precursor_mzs = []
    charges = []
    adducts = []
    smiles_list = []
    
    for i, (spec, mol) in enumerate(zip(spectra_data, mol_data)):
        # Process spectrum
        peaks = spec['peaks']
        if len(peaks) > 0:
            mz_array = np.array([p[0] for p in peaks], dtype=np.float32)
            intensity_array = np.array([p[1] for p in peaks], dtype=np.float32)
            
            # Take top peaks by intensity
            if len(peaks) > max_peaks:
                top_indices = np.argpartition(intensity_array, -max_peaks)[-max_peaks:]
                mz_array = mz_array[top_indices]
                intensity_array = intensity_array[top_indices]
            
            # Sort by m/z
            sort_indices = np.argsort(mz_array)
            mz_array = mz_array[sort_indices]
            intensity_array = intensity_array[sort_indices]
            
            # Fill spectrum array
            n_peaks = len(mz_array)
            spectrum_array[i, 0, :n_peaks] = mz_array
            spectrum_array[i, 1, :n_peaks] = intensity_array
        
        # Collect metadata
        precursor_mzs.append(spec.get('precursor_mz', 0.0) or 0.0)
        charges.append(spec.get('charge', 1) or 1)
        adducts.append(spec.get('adduct', '[M+H]+') or '[M+H]+')
        smiles_list.append(mol.get('smiles', '') or '')
    
    # Compute fingerprints
    fingerprints = compute_morgan_simple(smiles_list)
    
    # Create fold assignments for cross-validation (80% train, 20% val)
    num_spectra = len(spectra_data)
    fold_assignments = ['train'] * int(0.8 * num_spectra) + ['val'] * (num_spectra - int(0.8 * num_spectra))
    np.random.seed(42)  # For reproducible splits
    np.random.shuffle(fold_assignments)
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('spectrum', data=spectrum_array, compression='gzip')
        f.create_dataset('precursor_mz', data=np.array(precursor_mzs, dtype=np.float32), compression='gzip')
        f.create_dataset('charge', data=np.array(charges, dtype=np.int32), compression='gzip')
        f.create_dataset('adduct', data=[s.encode('utf-8') for s in adducts], compression='gzip')
        f.create_dataset('smiles', data=[s.encode('utf-8') for s in smiles_list], compression='gzip')
        f.create_dataset('fp_morgan_4096', data=fingerprints, compression='gzip')
        f.create_dataset('fold', data=[s.encode('utf-8') for s in fold_assignments], compression='gzip')
        
        # Metadata
        f.attrs['dataset'] = 'CANOPUS'
        f.attrs['num_spectra'] = num_spectra
        f.attrs['max_peaks'] = max_peaks
        f.attrs['converted_from'] = 'Simple'
        
        print(f"Successfully created HDF5 file with {num_spectra} spectra")
        print(f"Fold distribution: {pd.Series(fold_assignments).value_counts().to_dict()}")
        print(f"Columns: {list(f.keys())}")

def main():
    parser = argparse.ArgumentParser(description="Simple CANOPUS to DreaMS converter")
    parser.add_argument("canopus_dir", help="Path to CANOPUS dataset directory")
    parser.add_argument("output_path", help="Output HDF5 file path")
    
    args = parser.parse_args()
    
    try:
        # Load data
        spectra_data, mol_data = load_canopus_simple(args.canopus_dir)
        
        # Convert to HDF5
        create_dreams_hdf5_simple(spectra_data, mol_data, args.output_path)
        
        print(f"\n✅ Conversion completed successfully!")
        print(f"Output file: {args.output_path}")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
