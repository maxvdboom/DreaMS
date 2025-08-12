#!/usr/bin/env python3
"""
fix_hdf5_spectra.py

Fix existing HDF5 dataset by removing/correcting invalid spectra that cause division by zero errors.
"""

import h5py
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

def check_and_fix_hdf5(hdf5_path, output_path=None):
    """Check and fix spectra in HDF5 file."""
    if output_path is None:
        output_path = hdf5_path.replace('.hdf5', '_fixed.hdf5')
    
    print(f"Checking HDF5 file: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f_in:
        num_spectra = f_in['spectrum'].shape[0]
        print(f"Total spectra: {num_spectra}")
        
        # Check for problematic spectra
        problematic_indices = []
        fixed_spectra = []
        
        for i in tqdm(range(num_spectra), desc="Checking spectra"):
            spectrum = f_in['spectrum'][i]  # Shape: (2, max_peaks)
            mz_values = spectrum[0]
            intensity_values = spectrum[1]
            
            # Check for issues
            has_issues = False
            issue_reasons = []
            
            # Check for all-zero intensities
            if intensity_values.max() <= 0:
                has_issues = True
                issue_reasons.append("all_zero_intensities")
            
            # Check for NaN or infinite values
            if not np.isfinite(intensity_values).all() or not np.isfinite(mz_values).all():
                has_issues = True
                issue_reasons.append("nan_or_inf_values")
            
            # Check for negative intensities
            if (intensity_values < 0).any():
                has_issues = True
                issue_reasons.append("negative_intensities")
            
            if has_issues:
                problematic_indices.append(i)
                print(f"Spectrum {i} has issues: {issue_reasons}")
                
                # Fix the spectrum
                fixed_spectrum = fix_spectrum(spectrum)
                fixed_spectra.append((i, fixed_spectrum))
            
        print(f"Found {len(problematic_indices)} problematic spectra")
        
        if len(problematic_indices) > 0:
            print(f"Creating fixed dataset: {output_path}")
            
            # Create fixed dataset
            with h5py.File(output_path, 'w') as f_out:
                # Copy all datasets
                for key in f_in.keys():
                    if key == 'spectrum':
                        # Copy spectrum data with fixes
                        spectrum_data = f_in[key][:]
                        for idx, fixed_spectrum in fixed_spectra:
                            spectrum_data[idx] = fixed_spectrum
                        f_out.create_dataset('spectrum', data=spectrum_data, compression='gzip')
                    else:
                        # Copy other datasets as-is
                        f_out.create_dataset(key, data=f_in[key][:], compression='gzip')
                
                # Copy attributes
                for key, value in f_in.attrs.items():
                    f_out.attrs[key] = value
                
                f_out.attrs['fixed_spectra'] = len(problematic_indices)
            
            print(f"✅ Fixed dataset saved to: {output_path}")
        else:
            print("✅ No issues found in the dataset")
    
    return len(problematic_indices)

def fix_spectrum(spectrum):
    """Fix a problematic spectrum."""
    mz_values = spectrum[0].copy()
    intensity_values = spectrum[1].copy()
    
    # Remove NaN and infinite values
    valid_mask = np.isfinite(mz_values) & np.isfinite(intensity_values) & (mz_values > 0) & (intensity_values > 0)
    
    if valid_mask.any():
        # Keep only valid peaks
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        
        # Reset spectrum
        fixed_spectrum = np.zeros_like(spectrum)
        fixed_spectrum[0, :n_valid] = mz_values[valid_indices]
        fixed_spectrum[1, :n_valid] = intensity_values[valid_indices]
    else:
        # No valid peaks - create minimal valid spectrum
        fixed_spectrum = np.zeros_like(spectrum)
        fixed_spectrum[0, 0] = 100.0  # Dummy m/z
        fixed_spectrum[1, 0] = 1.0    # Dummy intensity
    
    return fixed_spectrum

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 fix_hdf5_spectra.py <hdf5_file> [output_file]")
        sys.exit(1)
    
    hdf5_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(hdf5_path).exists():
        print(f"File not found: {hdf5_path}")
        sys.exit(1)
    
    num_issues = check_and_fix_hdf5(hdf5_path, output_path)
    
    if num_issues > 0:
        print(f"\n🔧 Fixed {num_issues} problematic spectra")
        print("Use the fixed dataset for training to avoid division by zero errors")
    else:
        print("\n✅ Dataset is clean - no fixes needed")

if __name__ == "__main__":
    main()
