#!/bin/bash
#SBATCH -A m4431_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH -t 00:15:00
#SBATCH -J chestxray_8gpu
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -eo pipefail

module load pytorch
cd "$SCRATCH/DDL_ChestXray"

mkdir -p logs results checkpoints

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"

echo "Sampler: ${SAMPLER:-default}"
echo "Epochs: ${EPOCHS:-1}"
echo "Seed: ${SEED:-0}"
echo "Batch per GPU: ${BATCH:-64}"
echo "LR scaling: ${LRSCALE:-linear}"
echo "Warmup epochs: ${WARMUP:-1}"
echo "Max train: ${MAXTRAIN:-0}"
echo "Max val: ${MAXVAL:-0}"

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --kill-on-bad-exit=1 bash -c '

    set -eo pipefail

    echo "Starting node $(hostname), rank $SLURM_NODEID"

    nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,memory.used \
        --format=csv -l 5 \
        > "logs/gpu_util_${SLURM_JOB_ID}_$(hostname).csv" &

    SMI_PID=$!
    trap "kill $SMI_PID 2>/dev/null || true" EXIT

    torchrun \
        --nnodes=2 \
        --nproc_per_node=4 \
        --node_rank="$SLURM_NODEID" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        scripts/train_ddp.py \
        --epochs "${EPOCHS:-1}" \
        --seed "${SEED:-0}" \
        --batch-size "${BATCH:-64}" \
        --workers "${WORKERS:-8}" \
        --lr 1e-4 \
        --lr-scale "${LRSCALE:-linear}" \
        --warmup-epochs "${WARMUP:-1}" \
        --sampler "${SAMPLER:-default}" \
        --rare-per-batch "${RPB:-8}" \
        --max-train "${MAXTRAIN:-0}" \
        --max-val "${MAXVAL:-0}"
'

echo "End time: $(date)"

