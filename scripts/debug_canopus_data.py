#!/usr/bin/env python3
"""
debug_canopus_data.py

Debug script to inspect CANOPUS dataset structure and understand why no spectra are being loaded.
"""

import sys
from pathlib import Path
import pandas as pd

def inspect_canopus_directory(canopus_dir):
    """Inspect the CANOPUS directory structure."""
    canopus_dir = Path(canopus_dir)
    
    print(f"Inspecting CANOPUS directory: {canopus_dir}")
    print(f"Directory exists: {canopus_dir.exists()}")
    
    if not canopus_dir.exists():
        return
    
    print("\nDirectory contents:")
    for item in sorted(canopus_dir.iterdir()):
        if item.is_dir():
            print(f"  📁 {item.name}/")
            # List subdirectory contents
            try:
                sub_items = list(item.iterdir())[:10]  # Limit to first 10 items
                for sub_item in sub_items:
                    print(f"    📄 {sub_item.name}")
                if len(list(item.iterdir())) > 10:
                    print(f"    ... and {len(list(item.iterdir())) - 10} more items")
            except PermissionError:
                print(f"    (Permission denied)")
        else:
            size = item.stat().st_size if item.exists() else 0
            print(f"  📄 {item.name} ({size} bytes)")
    
    # Check for labels file
    labels_file = canopus_dir / "labels.tsv"
    if labels_file.exists():
        print(f"\n📊 Labels file found: {labels_file}")
        try:
            df = pd.read_csv(labels_file, sep="\t")
            print(f"Labels file shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"First few rows:")
            print(df.head())
        except Exception as e:
            print(f"Error reading labels file: {e}")
    else:
        print(f"\n❌ Labels file not found: {labels_file}")
    
    # Look for .ms files
    print(f"\n🔍 Searching for .ms files...")
    ms_files = list(canopus_dir.rglob("*.ms"))
    print(f"Found {len(ms_files)} .ms files")
    
    if ms_files:
        print("Sample .ms files:")
        for ms_file in ms_files[:5]:  # Show first 5
            print(f"  📄 {ms_file}")
            try:
                # Try to read first few lines
                with open(ms_file, 'r') as f:
                    lines = f.readlines()[:5]
                    print(f"    First few lines: {[line.strip() for line in lines]}")
            except Exception as e:
                print(f"    Error reading file: {e}")
    else:
        print("❌ No .ms files found!")
        
        # Look for other spectral file formats
        other_formats = ['.mgf', '.msp', '.mzml', '.mzxml', '.txt']
        for ext in other_formats:
            files = list(canopus_dir.rglob(f"*{ext}"))
            if files:
                print(f"Found {len(files)} {ext} files:")
                for f in files[:3]:
                    print(f"  📄 {f}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 debug_canopus_data.py <canopus_directory>")
        sys.exit(1)
    
    canopus_dir = sys.argv[1]
    inspect_canopus_directory(canopus_dir)

if __name__ == "__main__":
    main()
