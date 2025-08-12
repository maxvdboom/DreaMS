#!/usr/bin/env python3
"""
check_environment.py

Check if DreaMS environment variables are properly set up.
"""

import sys
from pathlib import Path
import os

def check_dreams_environment():
    """Check DreaMS environment setup."""
    print("🔍 Checking DreaMS environment setup...")
    
    try:
        # Add DreaMS to path
        dreams_path = Path(__file__).parent.parent
        sys.path.append(str(dreams_path))
        
        from dreams.definitions import PRETRAINED, DATA_DIR, DREAMS_DIR
        
        print(f"✅ DreaMS imported successfully")
        print(f"📁 DREAMS_DIR: {DREAMS_DIR}")
        print(f"📁 DATA_DIR: {DATA_DIR}")
        print(f"📁 PRETRAINED: {PRETRAINED}")
        
        # Check if pretrained directory exists
        if PRETRAINED.exists():
            print(f"✅ Pretrained models directory exists")
            
            # Check for SSL model
            ssl_model_path = PRETRAINED / "ssl_model.ckpt"
            if ssl_model_path.exists():
                print(f"✅ SSL model found: {ssl_model_path}")
            else:
                print(f"⚠️  SSL model not found: {ssl_model_path}")
                print("   You may need to download it first")
        else:
            print(f"⚠️  Pretrained models directory doesn't exist: {PRETRAINED}")
            print("   Creating directory...")
            PRETRAINED.mkdir(parents=True, exist_ok=True)
        
        # Check environment variables
        print(f"\n🌍 Environment variables:")
        env_vars = ['PRETRAINED', 'DATA_DIR', 'DREAMS_DIR']
        
        # Export environment variables
        from dreams.definitions import export
        export()
        
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                print(f"✅ {var}: {value}")
            else:
                print(f"⚠️  {var}: Not set")
        
        # Check dataset path
        print(f"\n📊 Checking dataset...")
        dataset_path = Path("./data/paired_spectra/canopus_train_dreams.hdf5")
        if dataset_path.exists():
            print(f"✅ Dataset found: {dataset_path}")
            
            # Quick HDF5 check
            import h5py
            try:
                with h5py.File(dataset_path, 'r') as f:
                    print(f"✅ HDF5 file is valid")
                    print(f"   Columns: {list(f.keys())}")
                    if 'fold' in f.keys():
                        print(f"✅ 'fold' column present")
                    else:
                        print(f"❌ 'fold' column missing - reconvert dataset")
            except Exception as e:
                print(f"❌ HDF5 file error: {e}")
        else:
            print(f"⚠️  Dataset not found: {dataset_path}")
            print("   Convert your CANOPUS dataset first")
        
        print(f"\n✅ Environment check completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import DreaMS: {e}")
        return False
    except Exception as e:
        print(f"❌ Environment check failed: {e}")
        return False

if __name__ == "__main__":
    success = check_dreams_environment()
    if not success:
        sys.exit(1)
