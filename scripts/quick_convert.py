#!/usr/bin/env python3
"""
Quick batch converter for your hypermarker data.

This script is a simple wrapper around the main batch converter
specifically configured for your data structure.
"""

import sys
import os
from pathlib import Path

# Add the scripts directory to Python path so we can import the main converter
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from batch_mzml_to_hdf5 import process_batch_directory

def main():
    if len(sys.argv) != 2:
        print("Usage: python quick_convert.py <batch_directory_name>")
        print("\nExample:")
        print("  python quick_convert.py batch01")
        print("  python quick_convert.py batch02")
        print("\nThis will process: /Users/maxvandenboom/Docs/Coding/AI/active/data/hypermarker/formatted/<batch_name>")
        print("Output will go to: hdf5/<batch_name>/")
        sys.exit(1)
    
    batch_name = sys.argv[1]
    
    # Construct the full path
    base_path = "/Users/maxvandenboom/Docs/Coding/AI/active/data/hypermarker/formatted"
    batch_path = os.path.join(base_path, batch_name)
    
    print(f"Processing batch: {batch_name}")
    print(f"Input directory: {batch_path}")
    print(f"Output directory: hdf5/{batch_name}/")
    print("-" * 50)
    
    # Process the batch
    successful, failed = process_batch_directory(batch_path)
    
    # Print summary
    print("\n" + "="*50)
    print("CONVERSION SUMMARY")
    print("="*50)
    print(f"Successfully converted: {successful} files")
    print(f"Failed conversions: {failed} files")
    print(f"Total files processed: {successful + failed}")
    
    if failed > 0:
        print(f"\nWarning: {failed} files failed to convert.")
        sys.exit(1)
    else:
        print("\nAll files converted successfully!")

if __name__ == "__main__":
    main()
