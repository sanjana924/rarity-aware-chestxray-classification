import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import roc_auc_score

from dataset import ChestXrayDataset, LABELS
from model import create_model
from rarity_sampler import RarityAwareSampler


def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)

            all_labels.append(labels.numpy())
            all_scores.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_scores)

    aucs = {}
    for i, name in enumerate(LABELS):
        if len(np.unique(y_true[:, i])) == 2:
            aucs[name] = float(
                roc_auc_score(y_true[:, i], y_score[:, i])
            )
        else:
            aucs[name] = float("nan")

    valid = [v for v in aucs.values() if np.isfinite(v)]
    return aucs, float(np.mean(valid)) if valid else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train", type=int, default=128)
    parser.add_argument("--max-val", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scale", choices=["linear", "sqrt", "none"], default="linear")
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--sampler", choices=["default", "rarity"], default="default")
    parser.add_argument("--rare-per-batch", type=int, default=8)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main = rank == 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.lr_scale == "sqrt":
        lr = args.lr * world_size ** 0.5
    elif args.lr_scale == "none":
        lr = args.lr
    else:
        lr = args.lr * world_size

    tag = f"{world_size}gpu_{args.sampler}"
    if args.sampler == "rarity" and args.rare_per_batch != 8:
        tag += f"_rpb{args.rare_per_batch}"
    if args.batch_size != 64:
        tag += f"_bs{args.batch_size}"
    if args.lr_scale != "linear":
        tag += f"_{args.lr_scale}"
    tag += f"_seed{args.seed}"

    if is_main:
        print("GPU:", torch.cuda.get_device_name(local_rank), flush=True)
        print(f"GPUs: {world_size}, per-GPU batch: {args.batch_size}, "
              f"global batch: {args.batch_size * world_size}, lr: {lr}, "
              f"lr scale: {args.lr_scale}, warmup epochs: {args.warmup_epochs}, "
              f"sampler: {args.sampler}, rare per batch: {args.rare_per_batch}, "
              f"epochs: {args.epochs}, seed: {args.seed}, workers: {args.workers}", flush=True)

    train_data = ChestXrayDataset("train")
    val_data = ChestXrayDataset("val")
    train_df = train_data.df

    if args.max_train > 0:
        n = min(args.max_train, len(train_data))
        train_data = Subset(train_data, range(n))
        train_df = train_df.iloc[:n]
    if args.max_val > 0:
        val_data = Subset(
            val_data, range(min(args.max_val, len(val_data)))
        )

    if args.sampler == "rarity":
        train_sampler = RarityAwareSampler(
            train_df, args.batch_size, args.rare_per_batch,
            num_replicas=world_size, rank=rank, seed=args.seed
        )
    else:
        train_sampler = DistributedSampler(
            train_data, num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed
        )

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True
    )

    warmup_steps = int(args.warmup_epochs * len(train_loader))

    model = create_model(pretrained=True)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DDP(model, device_ids=[local_rank])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
    )
    scaler = torch.amp.GradScaler("cuda")

    if is_main:
        print(f"Steps per epoch: {len(train_loader)}, warmup steps: {warmup_steps}", flush=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        with open(f"results/history_{tag}.csv", "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "loss", "train_seconds", "images_per_sec", "gpu_hours", "val_auroc"]
            )

    best_auc = -1.0
    total_seconds = 0.0

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        count = 0

        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * len(images)
            count += len(images)

        torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - start
        total_seconds += elapsed

        stats = torch.tensor([total_loss, count], device=device, dtype=torch.float64)
        dist.all_reduce(stats)
        total_loss, count = stats[0].item(), stats[1].item()

        if is_main:
            aucs, mean_auc = evaluate(model.module, val_loader, device)
            print(
                f"Epoch {epoch+1}: "
                f"loss={total_loss/count:.4f}, "
                f"train_seconds={elapsed:.2f}, "
                f"images_per_sec={count/elapsed:.2f}, "
                f"gpu_hours={elapsed * world_size / 3600:.4f}, "
                f"val_auroc={mean_auc:.4f}",
                flush=True
            )

            with open(f"results/history_{tag}.csv", "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch + 1, total_loss / count, elapsed,
                    count / elapsed, elapsed * world_size / 3600, mean_auc
                ])

            if mean_auc > best_auc:
                best_auc = mean_auc
                with open(f"results/validation_auc_{tag}.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["disease", "auroc"])
                    for name, auc in aucs.items():
                        writer.writerow([name, auc])

                torch.save(
                    model.module.state_dict(),
                    f"checkpoints/densenet121_{tag}.pt"
                )
                print(f"Saved best checkpoint (epoch {epoch+1})", flush=True)

        dist.barrier()

    if is_main:
        print(
            f"Training completed. best_val_auroc={best_auc:.4f}, "
            f"total_train_seconds={total_seconds:.2f}, "
            f"total_gpu_hours={total_seconds * world_size / 3600:.4f}",
            flush=True
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
