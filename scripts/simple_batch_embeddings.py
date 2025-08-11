"""
Simple Batch Embeddings Function

A simple function to process HDF5 files and generate DreaMS embeddings.
Can be imported and used in notebooks or other scripts.
"""

import os
import numpy as np
import h5py
import torch
from pathlib import Path
from dreams.api import dreams_embeddings


def validate_dreams_model():
    """
    Validate that the DreaMS model checkpoints are available and not corrupted.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        from dreams.api import PreTrainedModel
        from dreams.definitions import DREAMS_EMBEDDING
        
        # Try to initialize the model checkpoint path
        model_ckpt = DREAMS_EMBEDDING
        
        # Check if the checkpoint file exists
        if not os.path.exists(model_ckpt):
            return False, f"Model checkpoint not found: {model_ckpt}"
        
        # Try to load the checkpoint file to validate it's not corrupted
        try:
            torch.load(model_ckpt, map_location='cpu')
            return True, "Model checkpoint valid"
        except Exception as e:
            return False, f"Model checkpoint corrupted: {str(e)}"
            
    except ImportError as e:
        return False, f"DreaMS import error: {str(e)}"
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


def process_batch_embeddings(batch_dir_path, verbose=True):
    """
    Process all HDF5 files in a batch directory and generate embeddings.
    
    Args:
        batch_dir_path (str or Path): Path to the batch directory containing HDF5 files
        verbose (bool): Whether to print progress information
    
    Returns:
        dict: Dictionary mapping HDF5 file paths to their output embedding file paths
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
    
    if verbose:
        print(f"Embeddings will be saved to: {embs_dir}")
    
    # Validate DreaMS model first
    if verbose:
        print("\nValidating DreaMS model...")
    model_valid, model_error = validate_dreams_model()
    if not model_valid:
        raise RuntimeError(f"DreaMS model validation failed: {model_error}")
    if verbose:
        print("DreaMS model validation passed!")
    
    # Find all HDF5 files in the batch directory (recursively)
    hdf5_files = list(batch_dir.rglob("*.hdf5"))
    
    if not hdf5_files:
        if verbose:
            print(f"No HDF5 files found in {batch_dir}")
        return {}
    
    if verbose:
        print(f"Found {len(hdf5_files)} HDF5 files to process")
    
    # Validate files first
    valid_files = []
    invalid_files = []
    
    if verbose:
        print("\nValidating HDF5 files...")
    
    for hdf5_file in hdf5_files:
        is_valid, error_msg = validate_hdf5_file(hdf5_file)
        if is_valid:
            valid_files.append(hdf5_file)
        else:
            invalid_files.append((hdf5_file, error_msg))
            if verbose:
                print(f"INVALID: {hdf5_file.name} - {error_msg}")
    
    if verbose:
        print(f"\nValidation complete:")
        print(f"  Valid files: {len(valid_files)}")
        print(f"  Invalid files: {len(invalid_files)}")
    
    if not valid_files:
        if verbose:
            print("No valid HDF5 files found to process!")
        return {}
    
    results = {}
    successful_processes = 0
    failed_processes = 0
    
    # Process each valid HDF5 file
    for i, hdf5_file in enumerate(valid_files, 1):
        try:
            if verbose:
                print(f"\n[{i}/{len(valid_files)}] Processing: {hdf5_file.name}")
            
            # Create output filename (replace .hdf5 with .npy)
            output_filename = hdf5_file.stem + ".npy"
            output_path = embs_dir / output_filename
            
            # Check if output file already exists
            if output_path.exists():
                if verbose:
                    print(f"Embeddings file already exists, skipping: {output_path}")
                results[str(hdf5_file)] = str(output_path)
                successful_processes += 1
                continue
            
            # Generate embeddings
            embs = dreams_embeddings(str(hdf5_file))
            
            # Save embeddings
            np.save(output_path, embs)
            
            results[str(hdf5_file)] = str(output_path)
            successful_processes += 1
            
            if verbose:
                print(f"Saved embeddings to: {output_path}")
                print(f"Embeddings shape: {embs.shape}")
            
        except Exception as e:
            failed_processes += 1
            if verbose:
                print(f"Error processing {hdf5_file}: {str(e)}")
                # Optionally show full traceback for debugging
                import traceback
                traceback.print_exc()
            continue
    
    if verbose:
        print(f"\nProcessing complete!")
        print(f"  Successfully processed: {successful_processes}")
        print(f"  Failed to process: {failed_processes}")
        print(f"  Invalid files skipped: {len(invalid_files)}")
        print(f"Embeddings saved to: {embs_dir}")
    
    return results


# Example usage:
if __name__ == "__main__":
    # Example usage
    batch_directory = "/Users/maxvandenboom/Docs/Coding/AI/active/data/hypermarker/formatted/batch01"
    results = process_batch_embeddings(batch_directory)
    print(f"Processed {len(results)} files successfully")
