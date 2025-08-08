#!/usr/bin/env python3
"""
Batch mzML to HDF5 Converter

This script processes all mzML files in a given batch directory and converts them
to HDF5 format using the DreaMS MSData.from_mzml() method.

Usage:
    python batch_mzml_to_hdf5.py <batch_directory_path>

Example:
    python batch_mzml_to_hdf5.py /path/to/hypermarker/formatted/batch01

Output:
    HDF5 files will be saved to: hdf5/{batch_directory_name}/
"""

import sys
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

# Import DreaMS components
try:
    from dreams.utils.data import MSData
except ImportError:
    print("Error: Could not import DreaMS. Please ensure DreaMS is installed and available.")
    sys.exit(1)


def convert_mzml_to_hdf5(mzml_path, output_dir):
    """
    Convert a single mzML file to HDF5 format and move to output directory.
    
    Args:
        mzml_path (Path): Path to the input mzML file
        output_dir (Path): Directory where the HDF5 file should be moved
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load mzML data (this automatically creates HDF5 in the same directory)
        print(f"Processing: {mzml_path.name}")
        msdata = MSData.from_mzml(str(mzml_path))
        
        # The HDF5 file is automatically created in the same directory as the mzML
        expected_hdf5_path = mzml_path.with_suffix('.hdf5')
        
        if not expected_hdf5_path.exists():
            print(f"Error: Expected HDF5 file not found: {expected_hdf5_path}")
            return False
        
        # Move the HDF5 file to the output directory
        output_path = output_dir / expected_hdf5_path.name
        shutil.move(str(expected_hdf5_path), str(output_path))
        print(f"Moved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {mzml_path.name}: {str(e)}")
        return False


def process_batch_directory(batch_dir_path, base_output_dir="hdf5"):
    """
    Process all mzML files in a batch directory.
    
    Args:
        batch_dir_path (str): Path to the batch directory containing mzML files
        base_output_dir (str): Base directory for HDF5 output files
    
    Returns:
        tuple: (successful_conversions, failed_conversions)
    """
    batch_path = Path(batch_dir_path)
    
    # Validate input directory
    if not batch_path.exists():
        print(f"Error: Batch directory does not exist: {batch_path}")
        return 0, 0
    
    if not batch_path.is_dir():
        print(f"Error: Path is not a directory: {batch_path}")
        return 0, 0
    
    # Get batch directory name
    batch_name = batch_path.name
    
    # Create output directory structure
    output_dir = Path(base_output_dir) / batch_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Find all mzML files
    mzml_files = list(batch_path.glob("*.mzML"))
    
    if not mzml_files:
        print(f"No mzML files found in {batch_path}")
        return 0, 0
    
    print(f"Found {len(mzml_files)} mzML files to process")
    
    # Process each mzML file
    successful = 0
    failed = 0
    
    for mzml_file in tqdm(mzml_files, desc="Converting files"):
        # Create output filename (replace .mzML with .hdf5)
        output_filename = mzml_file.stem + ".hdf5"
        final_output_path = output_dir / output_filename
        
        # Skip if output file already exists in the target directory
        if final_output_path.exists():
            print(f"Skipping {mzml_file.name} (HDF5 file already exists in output directory)")
            continue
        
        # Also check if HDF5 already exists in the source directory
        source_hdf5_path = mzml_file.with_suffix('.hdf5')
        if source_hdf5_path.exists():
            print(f"Found existing HDF5 for {mzml_file.name}, moving to output directory")
            try:
                shutil.move(str(source_hdf5_path), str(final_output_path))
                print(f"Moved existing file to: {final_output_path}")
                successful += 1
            except Exception as e:
                print(f"Error moving existing HDF5 file {source_hdf5_path.name}: {str(e)}")
                failed += 1
            continue
        
        # Convert file and move to output directory
        if convert_mzml_to_hdf5(mzml_file, output_dir):
            successful += 1
        else:
            failed += 1
    
    return successful, failed


def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description="Convert mzML files in a batch directory to HDF5 format using DreaMS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_mzml_to_hdf5.py /path/to/batch01
  python batch_mzml_to_hdf5.py /Users/maxvandenboom/Docs/Coding/AI/active/data/hypermarker/formatted/batch01
  
Output files will be saved to: hdf5/{batch_directory_name}/
        """
    )
    
    parser.add_argument(
        "batch_directory",
        help="Path to the batch directory containing mzML files"
    )
    
    parser.add_argument(
        "--output-dir",
        default="hdf5",
        help="Base output directory for HDF5 files (default: hdf5)"
    )
    
    args = parser.parse_args()
    
    # Process the batch directory
    print(f"Starting batch conversion...")
    print(f"Input directory: {args.batch_directory}")
    
    successful, failed = process_batch_directory(args.batch_directory, args.output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("CONVERSION SUMMARY")
    print("="*50)
    print(f"Successfully converted: {successful} files")
    print(f"Failed conversions: {failed} files")
    print(f"Total files processed: {successful + failed}")
    
    if failed > 0:
        print(f"\nWarning: {failed} files failed to convert. Check error messages above.")
        sys.exit(1)
    else:
        print("\nAll files converted successfully!")


if __name__ == "__main__":
    main()
