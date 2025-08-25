#!/bin/bash
# Ultra memory-optimized fine-tuning script with extreme optimizations

# WandB Configuration
export WANDB_ENTITY="SSL_MAC"
export WANDB_API_KEY="122c488a229c45237791eb8cb419d1bc2ecc577a"

# Memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0

# First, set up environment variables
$(python -c "from dreams.definitions import export; export()")

# Run the ultra memory-optimized fine-tuning
echo "Starting ultra memory-optimized fine-tuning with aggressive settings..."
python3 dreams/training/train.py \
 --project_name MorganFingerprints \
 --job_key "morgan_4096_ultra_memory_optimized" \
 --run_name "morgan_4096_ultra_memory_$(date +%Y%m%d_%H%M%S)" \
 --wandb_entity_name "${WANDB_ENTITY}" \
 --train_objective fp_morgan_4096 \
 --train_regime fine-tuning \
 --dataset_pth "./data/paired_spectra/canopus_train_dreams.hdf5" \
 --dformat A \
 --model DreaMS \
 --num_workers_data 1 \
 --lr 1e-4 \
 --batch_size 1 \
 --prec_intens 1.1 \
 --num_devices 2 \
 --max_epochs 100 \
 --log_every_n_steps 50 \
 --head_depth 1 \
 --seed 3407 \
 --train_precision 32 \
 --pre_trained_pth "${PRETRAINED}/ssl_model.ckpt" \
 --val_check_interval 0.5 \
 --max_peaks_n 100 \
 --save_top_k 1 \
 --val_frac 0.2 \
 --weight_decay 1e-5 \
 --gradient_clip_val 1.0

echo "Ultra memory-optimized training completed."
