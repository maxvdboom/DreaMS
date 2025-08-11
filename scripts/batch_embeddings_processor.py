#!/usr/bin/env python3
"""
Batch Embeddings Processor

This script processes HDF5 files in a batch directory, generates DreaMS embeddings,
and saves them to an embeddings directory structure.

Usage:
    python batch_embeddings_processor.py /path/to/batch/directory

The script will:
1. Find all HDF5 files in the batch directory (recursively)
2. Generate embeddings using dreams_embeddings()
3. Save embeddings to ../embs/{batch_directory_name}/ relative to the batch directory
4. Use the original HDF5 filename (with .npy extension) for the embeddings file
"""

import os
import sys
import numpy as np
import h5py
import torch
from pathlib import Path
from tqdm import tqdm
from dreams.api import dreams_embeddings


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


# def validate_dreams_model():
#     """
#     Validate that the DreaMS model checkpoints are available and not corrupted.
    
#     Returns:
#         tuple: (is_valid, error_message)
#     """
#     try:
#         # Try to load the model by calling dreams_embeddings with a dummy path
#         # This will trigger the model loading and validation
#         # We expect this to fail due to invalid file path, but not due to corrupted model
        
#         # First, let's check if we can import and access the model loading functions
#         from dreams.api import PreTrainedModel
#         from dreams.definitions import DREAMS_EMBEDDING
        
#         # Try to initialize the model checkpoint path
#         model_ckpt = DREAMS_EMBEDDING
        
#         # Check if the checkpoint file exists
#         if not os.path.exists(model_ckpt):
#             return False, f"Model checkpoint not found: {model_ckpt}"
        
#         # Try to load the checkpoint file to validate it's not corrupted
#         try:
#             torch.load(model_ckpt, map_location='cpu')
#             return True, "Model checkpoint valid"
#         except Exception as e:
#             return False, f"Model checkpoint corrupted: {str(e)}"
            
#     except ImportError as e:
#         return False, f"DreaMS import error: {str(e)}"
#     except Exception as e:
#         return False, f"Model validation failed: {str(e)}"

def process_batch_directory(batch_dir_path):
    """
    Process all HDF5 files in a batch directory and generate embeddings.
    
    Args:
        batch_dir_path (str or Path): Path to the batch directory containing HDF5 files
    """
    batch_dir = Path(batch_dir_path)
    
    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch directory does not exist: {batch_dir}")
    
    if not batch_dir.is_dir():
        raise ValueError(f"Path is not a directory: {batch_dir}")
    
    # Get batch directory name for the embeddings folder
    batch_name = batch_dir.name
    
    # Create embeddings directory path (one level up from batch directory)
    embs_dir = batch_dir.parent / "embs" / batch_name
    
    # Create embeddings directory if it doesn't exist
    embs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Embeddings will be saved to: {embs_dir}")
    
    # Validate DreaMS model first
    # print("\nValidating DreaMS model...")
    # model_valid, model_error = validate_dreams_model()
    # if not model_valid:
    #     raise RuntimeError(f"DreaMS model validation failed: {model_error}")
    # print("DreaMS model validation passed!")
    
    # Find all HDF5 files in the batch directory (recursively)
    hdf5_files = list(batch_dir.rglob("*.hdf5"))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {batch_dir}")
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
            
            # Check if output file already exists
            output_filename = hdf5_file.stem + ".npy"
            output_path = embs_dir / output_filename
            
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
    print(f"Embeddings saved to: {embs_dir}")


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python batch_embeddings_processor.py /path/to/batch/directory")
        print("\nExample:")
        print("python batch_embeddings_processor.py /Users/maxvandenboom/Docs/Coding/AI/active/data/hypermarker/formatted/batch01")
        sys.exit(1)
    
    batch_directory = sys.argv[1]
    
    try:
        process_batch_directory(batch_directory)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
