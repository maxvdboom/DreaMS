#!/usr/bin/env python3
"""
DreaMS Model Diagnostics

This script helps diagnose issues with DreaMS model checkpoints and provides
options to re-download or fix corrupted model files.

Usage:
    python dreams_model_diagnostics.py
"""

import os
import sys
import torch
from pathlib import Path


def check_dreams_installation():
    """Check if DreaMS is properly installed and accessible."""
    try:
        import dreams
        from dreams.api import dreams_embeddings, PreTrainedModel
        from dreams.definitions import DREAMS_EMBEDDING
        print("✓ DreaMS import successful")
        return True, DREAMS_EMBEDDING
    except ImportError as e:
        print(f"✗ DreaMS import failed: {e}")
        return False, None


def check_model_file(model_path):
    """Check if the model checkpoint file exists and is valid."""
    print(f"\nChecking model file: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"✗ Model file does not exist: {model_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(model_path)
    print(f"✓ Model file exists, size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    
    if file_size == 0:
        print("✗ Model file is empty!")
        return False
    
    if file_size < 1024 * 1024:  # Less than 1MB is probably too small
        print("⚠ Warning: Model file seems unusually small")
    
    # Try to load the model file
    try:
        print("Testing model file integrity...")
        checkpoint = torch.load(model_path, map_location='cpu')
        print("✓ Model file loads successfully")
        
        # Check checkpoint structure
        if isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
            print(f"✓ Checkpoint contains {len(keys)} keys: {keys[:5]}{'...' if len(keys) > 5 else ''}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load model file: {e}")
        return False


def check_dreams_functionality():
    """Test if DreaMS can actually generate embeddings."""
    try:
        print("\nTesting DreaMS functionality with a dummy call...")
        # This should fail gracefully due to non-existent file, not due to model issues
        from dreams.api import dreams_embeddings
        dreams_embeddings("/nonexistent/file.hdf5")
        
    except FileNotFoundError:
        print("✓ DreaMS functions correctly (expected FileNotFoundError for dummy file)")
        return True
    except RuntimeError as e:
        if "PytorchStreamReader" in str(e) or "central directory" in str(e):
            print(f"✗ DreaMS model corruption detected: {e}")
            return False
        else:
            print(f"✓ DreaMS functions correctly (expected error: {e})")
            return True
    except Exception as e:
        print(f"⚠ Unexpected error: {e}")
        return False


def suggest_fixes():
    """Suggest potential fixes for model issues."""
    print("\n" + "="*60)
    print("POTENTIAL FIXES:")
    print("="*60)
    
    print("\n1. Re-download DreaMS model:")
    print("   - The model checkpoint may be corrupted")
    print("   - Try reinstalling DreaMS or manually downloading the model")
    
    print("\n2. Check disk space:")
    print("   - Ensure sufficient disk space for model files")
    print("   - Model files can be several hundred MB")
    
    print("\n3. Check file permissions:")
    print("   - Ensure read access to model checkpoint files")
    
    print("\n4. Clear cache and re-download:")
    print("   - Remove cached model files and let DreaMS re-download")
    
    print("\n5. Reinstall DreaMS:")
    print("   pip uninstall dreams")
    print("   pip install dreams")


def main():
    """Main diagnostic function."""
    print("DreaMS Model Diagnostics")
    print("="*40)
    
    # Check DreaMS installation
    dreams_ok, model_path = check_dreams_installation()
    if not dreams_ok:
        print("\nDreaMS is not properly installed or accessible.")
        return
    
    # Check model file
    model_ok = check_model_file(model_path)
    
    # Test functionality
    func_ok = check_dreams_functionality()
    
    # Summary
    print("\n" + "="*40)
    print("DIAGNOSTIC SUMMARY:")
    print("="*40)
    print(f"DreaMS Installation: {'✓ OK' if dreams_ok else '✗ FAILED'}")
    print(f"Model File: {'✓ OK' if model_ok else '✗ FAILED'}")
    print(f"DreaMS Functionality: {'✓ OK' if func_ok else '✗ FAILED'}")
    
    if not (model_ok and func_ok):
        suggest_fixes()
    else:
        print("\n✓ All checks passed! DreaMS should work correctly.")


if __name__ == "__main__":
    main()
