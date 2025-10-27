#!/usr/bin/env python3
"""
Batch Embeddings Processor

This script processes HDF5 files in an input directory, generates DreaMS embeddings,
and saves them to a specified output directory.

Usage:
    python batch_embeddings_processor.py /path/to/input/directory /path/to/output/directory

The script will:
1. Find all HDF5 files in the input directory (recursively)
2. Generate embeddings using dreams_embeddings()
3. Save embeddings to the output directory
4. Use the original HDF5 filename (with .npy extension) for the embeddings file
5. Preserve directory structure from input to output
"""

import os
import sys
import numpy as np
import h5py
import torch
from pathlib import Path
from tqdm import tqdm
from dreams.api import dreams_embeddings
from dreams.definitions import DREAMS_EMBEDDING, PRETRAINED


def validate_model_checkpoint():
    """
    Validate that both DreaMS model checkpoints are not corrupted.
    If corrupted, delete them so they can be re-downloaded.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Expected file sizes from Zenodo
        expected_sizes = {
            'embedding_model.ckpt': 1241290418,  # ~1.15 GB
            'ssl_model.ckpt': 1392710212,  # ~1.33 GB
        }
        
        messages = []
        for model_name, expected_size in expected_sizes.items():
            model_path = PRETRAINED / model_name
            
            if not model_path.exists():
                messages.append(f"{model_name}: not found (will be downloaded)")
                continue
            
            actual_size = model_path.stat().st_size
            
            # Check if file size matches expected size (allow 1% tolerance)
            size_tolerance = expected_size * 0.01
            if abs(actual_size - expected_size) > size_tolerance:
                print(f"\n{model_name} has incorrect size:")
                print(f"  Expected: {expected_size:,} bytes ({expected_size / (1024**3):.2f} GB)")
                print(f"  Actual: {actual_size:,} bytes ({actual_size / (1024**3):.2f} GB)")
                print(f"  Deleting corrupted file: {model_path}")
                model_path.unlink()
                messages.append(f"{model_name}: corrupted (deleted, will be re-downloaded)")
                continue
            
            # Try to load the checkpoint to validate it's not corrupted
            try:
                torch.load(str(model_path), map_location='cpu', weights_only=False)
                messages.append(f"{model_name}: valid")
            except Exception as e:
                print(f"\n{model_name} failed to load: {str(e)}")
                print(f"  Deleting corrupted file: {model_path}")
                model_path.unlink()
                messages.append(f"{model_name}: corrupted (deleted, will be re-downloaded)")
        
        return True, "; ".join(messages)
            
    except Exception as e:
        return False, f"Model validation failed: {str(e)}"


def validate_hdf5_file(file_path):
    """
    Validate if an HDF5 file is readable and not corrupted.
    
    Args:
        file_path (Path): Path to the HDF5 file
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check if file exists and has reasonable size
        if not file_path.exists():
            return False, "File does not exist"
        
        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, "File is empty (0 bytes)"
        
        if file_size < 1024:  # Less than 1KB is probably too small for a valid HDF5
            return False, f"File too small ({file_size} bytes)"
        
        # Try to open the HDF5 file to check if it's valid
        with h5py.File(file_path, 'r') as f:
            # Try to list keys to ensure the file structure is accessible
            list(f.keys())
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"HDF5 validation failed: {str(e)}"


def process_batch_directory(input_dir_path, output_dir_path):
    """
    Process all HDF5 files in an input directory and generate embeddings.
    
    Args:
        input_dir_path (str or Path): Path to the input directory containing HDF5 files
        output_dir_path (str or Path): Path to the output directory for embeddings
    """
    input_dir = Path(input_dir_path)
    output_dir = Path(output_dir_path)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Validate DreaMS model first
    print("\nValidating DreaMS model checkpoint...")
    model_valid, model_msg = validate_model_checkpoint()
    if not model_valid:
        raise RuntimeError(f"DreaMS model validation failed: {model_msg}")
    print(f"Model validation: {model_msg}")
    
    # Find all HDF5 files in the input directory (recursively)
    hdf5_files = list(input_dir.rglob("*.hdf5"))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {input_dir}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files to process")
    
    # Validate files first
    valid_files = []
    invalid_files = []
    
    print("\nValidating HDF5 files...")
    for hdf5_file in hdf5_files:
        is_valid, error_msg = validate_hdf5_file(hdf5_file)
        if is_valid:
            valid_files.append(hdf5_file)
        else:
            invalid_files.append((hdf5_file, error_msg))
            print(f"INVALID: {hdf5_file.name} - {error_msg}")
    
    print(f"\nValidation complete:")
    print(f"  Valid files: {len(valid_files)}")
    print(f"  Invalid files: {len(invalid_files)}")
    
    if not valid_files:
        print("No valid HDF5 files found to process!")
        return
    
    # Process each valid HDF5 file
    successful_processes = 0
    failed_processes = 0
    
    for hdf5_file in tqdm(valid_files, desc="Processing HDF5 files"):
        try:
            # Generate embeddings
            print(f"\nProcessing: {hdf5_file.name}")
            
            # Preserve directory structure from input to output
            relative_path = hdf5_file.relative_to(input_dir)
            output_filename = relative_path.stem + ".npy"
            output_path = output_dir / relative_path.parent / output_filename
            
            # Create subdirectories if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.exists():
                print(f"Embeddings file already exists, skipping: {output_path}")
                successful_processes += 1
                continue
            
            embs = dreams_embeddings(str(hdf5_file))
            
            # Save embeddings
            np.save(output_path, embs)
            print(f"Saved embeddings to: {output_path}")
            print(f"Embeddings shape: {embs.shape}")
            successful_processes += 1
            
        except Exception as e:
            print(f"Error processing {hdf5_file}: {str(e)}")
            failed_processes += 1
            
            # Log the full error details for debugging
            import traceback
            print(f"Full error traceback:")
            traceback.print_exc()
            continue
    
    print(f"\nProcessing complete!")
    print(f"  Successfully processed: {successful_processes}")
    print(f"  Failed to process: {failed_processes}")
    print(f"  Invalid files skipped: {len(invalid_files)}")
    print(f"Embeddings saved to: {output_dir}")


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 3:
        print("Usage: python batch_embeddings_processor.py /path/to/input/directory /path/to/output/directory")
        print("\nExample:")
        print("python batch_embeddings_processor.py /Users/maxvandenboom/Docs/Coding/AI/active/data/hypermarker/formatted/batch01 /Users/maxvandenboom/Docs/Coding/AI/active/data/hypermarker/embeddings/batch01")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    try:
        process_batch_directory(input_directory, output_directory)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
