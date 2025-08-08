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
from pathlib import Path
from tqdm import tqdm
from dreams.api import dreams_embeddings


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
    
    # Find all HDF5 files in the batch directory (recursively)
    hdf5_files = list(batch_dir.rglob("*.hdf5"))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {batch_dir}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files to process")
    
    # Process each HDF5 file
    for hdf5_file in tqdm(hdf5_files, desc="Processing HDF5 files"):
        try:
            # Generate embeddings
            print(f"\nProcessing: {hdf5_file.name}")
            embs = dreams_embeddings(str(hdf5_file))
            
            # Create output filename (replace .hdf5 with .npy)
            output_filename = hdf5_file.stem + ".npy"
            output_path = embs_dir / output_filename
            
            # Save embeddings
            np.save(output_path, embs)
            print(f"Saved embeddings to: {output_path}")
            print(f"Embeddings shape: {embs.shape}")
            
        except Exception as e:
            print(f"Error processing {hdf5_file}: {str(e)}")
            continue
    
    print(f"\nProcessing complete! Embeddings saved to: {embs_dir}")


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
