#!/bin/bash
#SBATCH -A m4431_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 00:30:00
#SBATCH -J chestxray_eval
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load pytorch
cd $SCRATCH/DDL_ChestXray

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

srun python scripts/eval_metrics.py --workers 16 --max-val ${MAXVAL:-0} --max-test ${MAXTEST:-0}

echo "End time: $(date)"
