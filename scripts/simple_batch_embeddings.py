"""
Simple Batch Embeddings Function

A simple function to process HDF5 files and generate DreaMS embeddings.
Can be imported and used in notebooks or other scripts.
"""

import os
import numpy as np
from pathlib import Path
from dreams.api import dreams_embeddings


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
    
    # Find all HDF5 files in the batch directory (recursively)
    hdf5_files = list(batch_dir.rglob("*.hdf5"))
    
    if not hdf5_files:
        if verbose:
            print(f"No HDF5 files found in {batch_dir}")
        return {}
    
    if verbose:
        print(f"Found {len(hdf5_files)} HDF5 files to process")
    
    results = {}
    
    # Process each HDF5 file
    for i, hdf5_file in enumerate(hdf5_files, 1):
        try:
            if verbose:
                print(f"\n[{i}/{len(hdf5_files)}] Processing: {hdf5_file.name}")
            
            # Generate embeddings
            embs = dreams_embeddings(str(hdf5_file))
            
            # Create output filename (replace .hdf5 with .npy)
            output_filename = hdf5_file.stem + ".npy"
            output_path = embs_dir / output_filename
            
            # Save embeddings
            np.save(output_path, embs)
            
            results[str(hdf5_file)] = str(output_path)
            
            if verbose:
                print(f"Saved embeddings to: {output_path}")
                print(f"Embeddings shape: {embs.shape}")
            
        except Exception as e:
            if verbose:
                print(f"Error processing {hdf5_file}: {str(e)}")
            continue
    
    if verbose:
        print(f"\nProcessing complete! {len(results)} files processed successfully.")
        print(f"Embeddings saved to: {embs_dir}")
    
    return results


# Example usage:
if __name__ == "__main__":
    # Example usage
    batch_directory = "/Users/maxvandenboom/Docs/Coding/AI/active/data/hypermarker/formatted/batch01"
    results = process_batch_embeddings(batch_directory)
    print(f"Processed {len(results)} files successfully")
