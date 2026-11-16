#!/bin/bash
#SBATCH -A m4431_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 00:30:00
#SBATCH -J chestxray_1gpu
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load pytorch
cd $SCRATCH/DDL_ChestXray

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 5 > logs/gpu_util_${SLURM_JOB_ID}.csv &
SMI_PID=$!

srun torchrun --standalone --nproc_per_node=1 scripts/train_ddp.py \
    --max-train ${MAXTRAIN:-0} --max-val ${MAXVAL:-0} \
    --batch-size 16 --workers 4 --lr 1e-4 \
    --sampler ${SAMPLER:-default}

kill $SMI_PID
echo "End time: $(date)"
