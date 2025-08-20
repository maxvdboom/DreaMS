#!/usr/bin/env python3
"""
Modified fine-tuning command with memory optimizations.
This reduces memory usage while maintaining training effectiveness.
"""

# First, set up environment variables
$(python -c "from dreams.definitions import export; export()")

# Memory-optimized fine-tuning for Morgan 4096 fingerprints
echo "Starting memory-optimized fine-tuning..."
python3 dreams/training/train.py \
 --no_wandb \
 --project_name MorganFingerprints \
 --job_key "morgan_4096_fine_tune_mem_opt" \
 --run_name "morgan_4096_fine_tune_mem_opt" \
 --train_objective fp_morgan_4096 \
 --train_regime fine-tuning \
 --dataset_pth "./data/paired_spectra/canopus_train_dreams.hdf5" \
 --dformat A \
 --model DreaMS \
 --num_workers_data 2 \
 --lr 1e-4 \
 --batch_size 1 \
 --prec_intens 1.1 \
 --num_devices 1 \
 --max_epochs 100 \
 --log_every_n_steps 20 \
 --head_depth 1 \
 --seed 3407 \
 --train_precision 32 \
 --pre_trained_pth "${PRETRAINED}/ssl_model.ckpt" \
 --val_check_interval 0.5 \
 --max_peaks_n 50 \
 --save_top_k 1 \
 --val_frac 0.1 \
 --weight_decay 1e-5 \
 --no_val
