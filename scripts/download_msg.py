# Download the Complete MassSpecGym Dataset
import os
from tqdm import tqdm
import time
# Import necessary libraries
import pandas as pd
import numpy as np
from datasets import load_dataset
import os
from huggingface_hub import hf_hub_download, list_repo_files
import json


def download_complete_massspecgym(save_dir="../data/massspecgym_complete"):
    """
    Download and save the complete MassSpecGym dataset
    """
    print("🚀 Starting download of complete MassSpecGym dataset...")
    repo_id = "roman-bushuiev/MassSpecGym"
    dataset = load_dataset(repo_id)

    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all available splits
    available_splits = list(dataset.keys())
    print(f"Available splits: {available_splits}")
    
    total_samples = 0
    download_info = {}
    
    for split_name in available_splits:
        split_data = dataset[split_name]
        split_size = len(split_data)
        total_samples += split_size
        
        print(f"\n📊 Processing {split_name} split ({split_size:,} samples)...")
        
        # Save split to parquet files in chunks for memory efficiency
        chunk_size = 10000  # Process 10k samples at a time
        num_chunks = (split_size + chunk_size - 1) // chunk_size
        
        split_dir = os.path.join(save_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        chunk_files = []
        
        for chunk_idx in tqdm(range(num_chunks), desc=f"Downloading {split_name} chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, split_size)
            
            # Select chunk
            chunk_data = split_data.select(range(start_idx, end_idx))
            
            # Convert to pandas and save
            chunk_df = chunk_data.to_pandas()
            
            # Save chunk
            chunk_filename = f"{split_name}_chunk_{chunk_idx:04d}.parquet"
            chunk_path = os.path.join(split_dir, chunk_filename)
            chunk_df.to_parquet(chunk_path, compression='snappy')
            chunk_files.append(chunk_filename)
            
            # Memory cleanup
            del chunk_df, chunk_data
        
        download_info[split_name] = {
            'num_samples': split_size,
            'num_chunks': num_chunks,
            'chunk_files': chunk_files,
            'directory': split_dir
        }
        
        print(f"✅ {split_name} split saved: {num_chunks} chunks, {split_size:,} samples")
    
    # Save metadata
    metadata = {
        'dataset_name': 'MassSpecGym',
        'download_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': total_samples,
        'splits': download_info,
        'chunk_size': chunk_size,
        'compression': 'snappy'
    }
    
    import json
    metadata_path = os.path.join(save_dir, 'dataset_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n🎉 Download complete!")
    print(f"📁 Saved to: {save_dir}")
    print(f"📊 Total samples: {total_samples:,}")
    print(f"💾 Metadata saved to: {metadata_path}")
    
    # Calculate approximate disk usage
    total_size_mb = 0
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                total_size_mb += os.path.getsize(file_path) / (1024 * 1024)
    
    print(f"💽 Approximate disk usage: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
    
    return save_dir, metadata

# Start the download
print("⚠️  Warning: This will download the complete dataset which may be several GB in size.")
print("⚠️  Make sure you have sufficient disk space and a stable internet connection.")

user_confirm = input("\nDo you want to proceed with downloading the complete dataset? (y/N): ")

if user_confirm.lower() in ['y', 'yes']:
    # Automatically download the complete MassSpecGym dataset
    print("🚀 Starting automatic download of complete MassSpecGym dataset...")
    print("📊 This may take several minutes depending on your internet connection...")

    start_time = time.time()
    save_directory, metadata = download_complete_massspecgym()
    end_time = time.time()

    print(f"\n⏱️  Total download time: {(end_time - start_time)/60:.1f} minutes")
    print(f"🔗 Dataset ready for analysis at: {save_directory}")

    # Display final summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Dataset: MassSpecGym")
    print(f"Total samples: {metadata['total_samples']:,}")
    print(f"Splits: {list(metadata['splits'].keys())}")
    print(f"Storage location: {save_directory}")
    print(f"Download completed: {metadata['download_date']}")
    print("="*60)
    
    # print(f"\n⏱️  Total download time: {(end_time - start_time)/60:.1f} minutes")
    # print(f"🔗 Dataset ready for analysis at: {save_directory}")
else:
    print("❌ Download cancelled by user.")