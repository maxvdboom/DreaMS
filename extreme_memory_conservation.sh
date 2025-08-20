#!/bin/bash
# Alternative approach using model parallelism and CPU offloading

# First, set up environment variables
$(python -c "from dreams.definitions import export; export()")

echo "Starting model-parallel fine-tuning with CPU offloading..."

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=4

# Fine-tuning with extreme memory conservation
python3 dreams/training/train.py \
 --no_wandb \
 --project_name MorganFingerprints \
 --job_key "morgan_4096_model_parallel" \
 --run_name "morgan_4096_model_parallel" \
 --train_objective fp_morgan_4096 \
 --train_regime fine-tuning \
 --dataset_pth "./data/paired_spectra/canopus_train_dreams.hdf5" \
 --dformat A \
 --model DreaMS \
 --num_workers_data 1 \
 --lr 5e-5 \
 --batch_size 1 \
 --prec_intens 1.1 \
 --num_devices 1 \
 --max_epochs 200 \
 --log_every_n_steps 50 \
 --head_depth 1 \
 --seed 3407 \
 --train_precision 16 \
 --pre_trained_pth "${PRETRAINED}/ssl_model.ckpt" \
 --val_check_interval 1.0 \
 --max_peaks_n 30 \
 --save_top_k 1 \
 --val_frac 0.05 \
 --weight_decay 1e-4 \
 --no_val
