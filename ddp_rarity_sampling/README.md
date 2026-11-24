# Multi-GPU Training and Rarity-Aware Sampling

## Overview
This folder contains the multi-GPU training code, the rarity-aware sampler, the evaluation script, the Slurm job scripts, and all logs and results for our DDP experiments on NERSC Perlmutter.

Everyone uses these scripts for the final runs, so results are directly comparable.

## Files
| File | Description |
|---|---|
| `scripts/train_ddp.py` | Multi-GPU training with PyTorch DDP. Supports default and rarity-aware sampling, seeds, learning-rate scaling, warmup, best-checkpoint saving and per-epoch history logs. |
| `scripts/rarity_sampler.py` | Rarity-aware sampler. Puts a fixed number of rare-finding images (Edema, Fibrosis, Pneumonia, Hernia) in every batch, split across GPUs. |
| `scripts/eval_metrics.py` | Evaluates every checkpoint in `checkpoints/` on the validation and test sets (AUROC, AUPRC, precision, recall, F1, accuracy). |
| `scripts/check_sampler.py` | Checks how many rare images appear per training step with each sampler. |
| `run_1gpu.sh`, `run_2gpu.sh`, `run_4gpu.sh` | Slurm job scripts for 1, 2 and 4 GPUs. Settings are passed with `--export` (see below). |
| `run_eval.sh` | Slurm job script for `eval_metrics.py`. |
| `pretrained/` | ImageNet DenseNet-121 weights. |
| `logs/` | Slurm output, error and GPU utilization logs for every run. |
| `results/` | Per-epoch history, validation AUROC and test metric CSVs. |
| `checkpoints/` | Best checkpoint for each run. |

`dataset.py` and `model.py` come from `preprocessing/` and must be copied into `scripts/` before running.

## Final Experiment Setup
| Setting | Value |
|---|---|
| Model | DenseNet-121, ImageNet pretrained |
| Images | 224 x 224 preprocessed images (`processed_images/`) |
| Batch size | 64 per GPU (global 64 / 128 / 256 / 512 on 1 / 2 / 4 / 8 GPUs) |
| Base learning rate | 0.0001 at 1 GPU, scaled with GPU count (rule set by the learning-rate test) |
| Warmup | 1 epoch |
| Epochs | 8, keeping the checkpoint with the best validation AUROC |
| Rarity-aware sampling | 8 rare-finding images per batch of 64 (12.5%) |
| Data loader workers | 8 per GPU |
| Seeds | 0, 1, 2 |
| Optimizer | AdamW, mixed precision |
| GPUs | NVIDIA A100 80GB |

## How to Run
1. Copy this folder to your scratch space and add `dataset.py` and `model.py`:
```bash
cp -r ddp_rarity_sampling $SCRATCH/DDL_ChestXray
cp preprocessing/dataset.py preprocessing/model.py $SCRATCH/DDL_ChestXray/scripts/
cd $SCRATCH/DDL_ChestXray
mkdir -p logs results checkpoints
```

2. Submit a run. Example (2 GPUs, rarity-aware sampling, seed 1):
```bash
sbatch -t 01:00:00 --export=ALL,SAMPLER=rarity,EPOCHS=8,SEED=1 run_2gpu.sh
```

3. Settings you can pass with `--export`:

| Variable | Default | Meaning |
|---|---|---|
| `SAMPLER` | `default` | `default` or `rarity` |
| `EPOCHS` | 1 | Number of epochs (use 8 for final runs) |
| `SEED` | 0 | Random seed (0, 1 or 2) |
| `BATCH` | 64 | Batch size per GPU |
| `RPB` | 8 | Rare images per batch (rarity-aware only) |
| `LRSCALE` | `linear` | `linear`, `sqrt` or `none` |
| `WARMUP` | 1 | Warmup length in epochs |
| `WORKERS` | 8 | Data loader workers per GPU |
| `MAXTRAIN`, `MAXVAL` | 0 | Limit images for quick tests (0 = all) |

4. Experiment-specific settings:
- Sampler ratio experiment (1 GPU, rarity-aware): `RPB=4` and `RPB=16`
- Fixed global batch experiment (global batch 256, `LRSCALE=none`): `BATCH=128` on 2 GPUs, `BATCH=64` on 4 GPUs, `BATCH=32` on 8 GPUs. For rarity-aware runs keep 12.5% rare: `RPB=16`, `RPB=8`, `RPB=4`.

## Output Files
Each run's files are named by GPU count, sampler and seed, for example `4gpu_rarity_seed2`. Extra settings are added to the name when they differ from the defaults (`_bs128`, `_rpb4`, `_none`, `_sqrt`).

| File | Contents |
|---|---|
| `logs/chestxray_<N>gpu_<jobid>.out` | Training log: settings, per-epoch loss, time, images/sec, GPU-hours, validation AUROC |
| `logs/gpu_util_<jobid>.csv` | GPU utilization every 5 seconds (nvidia-smi) |
| `results/history_<tag>.csv` | One row per epoch |
| `results/validation_auc_<tag>.csv` | Per-disease validation AUROC at the best epoch |
| `checkpoints/densenet121_<tag>.pt` | Best checkpoint |

## Setup Tests
### Speed test (1 GPU, 20,000 images, 1 epoch)
| Setting | Images/sec | GPU utilization |
|---|---|---|
| 8 workers, batch 16 | 214 | 25% |
| 16 workers, batch 16 | 227 | 33% |
| 8 workers, batch 64 | 608 | 39% |

More data loader workers barely changed throughput. Batch 64 was 2.8x faster, so the main bottleneck was per-step overhead from the small batch, not image loading. Logs: `logs/int_w*_b*.out`.

### Epoch test (1 GPU, full data, 20 epochs, default sampling)
Validation AUROC peaked at epoch 4 (0.827) and then fell to about 0.73 by epoch 20, while training loss kept dropping. The model starts overfitting after about 4 epochs, so final runs use 8 epochs and keep the best checkpoint. Results: `results/epoch_test_history_1gpu.csv`.

## Midterm Results (batch 16, 1 epoch)
Earlier 1-GPU and 2-GPU runs with batch 16 and one epoch are kept for reference in `logs/` and `results/` (files without a seed in the name). These used the original 1024 x 1024 images and the earlier version of the training script.
