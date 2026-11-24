#!/bin/bash
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:30:00
#SBATCH -J cxr_bench
#SBATCH -o preprocessing/benchmark_results/bench_%j.out
#SBATCH -e preprocessing/benchmark_results/bench_%j.err

set -e

cd /pscratch/sd/p/pravch1/chestxray_project

module load pytorch

export CUDA_VISIBLE_DEVICES=0

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"

cd preprocessing

python -u benchmark_1gpu.py \
    --epochs 1 \
    --batch-size 16 \
    --max-train 4096 \
    --max-val 128 \
    --workers 4

echo "End time: $(date)"
