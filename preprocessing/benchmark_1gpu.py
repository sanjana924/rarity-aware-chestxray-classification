
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

from dataset import ChestXrayDataset, LABELS
from model import create_model


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
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. Run inside a GPU job.")

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    train_data = ChestXrayDataset("train")
    val_data = ChestXrayDataset("val")

    if args.max_train > 0:
        train_data = Subset(
            train_data, range(min(args.max_train, len(train_data)))
        )
    if args.max_val > 0:
        val_data = Subset(
            val_data, range(min(args.max_val, len(val_data)))
        )

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True
    )

    model = create_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        count = 0

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
        elapsed = time.perf_counter() - start

        print(
            f"Epoch {epoch+1}: "
            f"loss={total_loss/count:.4f}, "
            f"train_seconds={elapsed:.2f}, "
            f"images_per_sec={count/elapsed:.2f}",
            flush=True
        )

        aucs, mean_auc = evaluate(model, val_loader, device)
        print(f"Validation mean AUROC: {mean_auc:.4f}", flush=True)

        with open("results/validation_auc.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["disease", "auroc"])
            for name, auc in aucs.items():
                writer.writerow([name, auc])

        torch.save(
            model.state_dict(),
            "checkpoints/densenet121_1gpu.pt"
        )

    print("Training completed.", flush=True)


if __name__ == "__main__":
    main()
