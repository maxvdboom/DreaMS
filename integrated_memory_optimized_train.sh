#!/bin/bash
# Memory-optimized fine-tuning script using the modified train.py

# First, set up environment variables
$(python -c "from dreams.definitions import export; export()")

# Run the memory-optimized fine-tuning
echo "Starting memory-optimized fine-tuning with integrated optimizations..."
python3 dreams/training/train.py \
 --no_wandb \
 --project_name MorganFingerprints \
 --job_key "morgan_4096_fine_tune_integrated" \
 --run_name "morgan_4096_fine_tune_integrated" \
 --train_objective fp_morgan_4096 \
 --train_regime fine-tuning \
 --dataset_pth "./data/paired_spectra/canopus_train_dreams.hdf5" \
 --dformat A \
 --model DreaMS \
 --num_workers_data 4 \
 --lr 1e-4 \
 --batch_size 2 \
 --prec_intens 1.1 \
 --num_devices 1 \
 --max_epochs 100 \
 --log_every_n_steps 10 \
 --head_depth 1 \
 --seed 3407 \
 --train_precision 32 \
 --pre_trained_pth "${PRETRAINED}/ssl_model.ckpt" \
 --val_check_interval 0.1 \
 --max_peaks_n 100 \
 --save_top_k 3 \
 --val_frac 0.2 \
 --weight_decay 1e-5
