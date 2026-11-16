
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train", type=int, default=128)
    parser.add_argument("--max-val", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sampler", choices=["default", "rarity"], default="default")
    parser.add_argument("--rare-per-batch", type=int, default=2)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main = rank == 0

    torch.manual_seed(42)
    np.random.seed(42)

    lr = args.lr * world_size
    tag = f"{world_size}gpu_{args.sampler}"

    if is_main:
        print("GPU:", torch.cuda.get_device_name(local_rank), flush=True)
        print(f"GPUs: {world_size}, per-GPU batch: {args.batch_size}, "
              f"global batch: {args.batch_size * world_size}, lr: {lr}, "
              f"sampler: {args.sampler}", flush=True)

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
            num_replicas=world_size, rank=rank, seed=42
        )
    else:
        train_sampler = DistributedSampler(
            train_data, num_replicas=world_size, rank=rank,
            shuffle=True, seed=42
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

    model = create_model(pretrained=True)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DDP(model, device_ids=[local_rank])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda")

    if is_main:
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

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

            total_loss += loss.item() * len(images)
            count += len(images)

        torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - start

        stats = torch.tensor([total_loss, count], device=device, dtype=torch.float64)
        dist.all_reduce(stats)
        total_loss, count = stats[0].item(), stats[1].item()

        if is_main:
            print(
                f"Epoch {epoch+1}: "
                f"loss={total_loss/count:.4f}, "
                f"train_seconds={elapsed:.2f}, "
                f"images_per_sec={count/elapsed:.2f}, "
                f"gpu_hours={elapsed * world_size / 3600:.4f}",
                flush=True
            )

            aucs, mean_auc = evaluate(model.module, val_loader, device)
            print(f"Validation mean AUROC: {mean_auc:.4f}", flush=True)

            with open(f"results/validation_auc_{tag}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["disease", "auroc"])
                for name, auc in aucs.items():
                    writer.writerow([name, auc])

            torch.save(
                model.module.state_dict(),
                f"checkpoints/densenet121_{tag}.pt"
            )

        dist.barrier()

    if is_main:
        print("Training completed.", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
