import argparse
import csv
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from dataset import ChestXrayDataset, LABELS
from model import create_model

RARE_LABELS = ["Edema", "Fibrosis", "Pneumonia", "Hernia"]


def predict(models, loader, device):
    all_labels = []
    all_scores = [[] for _ in models]

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            all_labels.append(labels.numpy())
            for i, model in enumerate(models):
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(images)
                all_scores[i].append(torch.sigmoid(logits).float().cpu().numpy())

    y_true = np.concatenate(all_labels)
    return y_true, [np.concatenate(s) for s in all_scores]


def best_thresholds(y_true, y_score):
    thresholds = []
    for i in range(len(LABELS)):
        if len(np.unique(y_true[:, i])) < 2:
            thresholds.append(0.5)
            continue
        p, r, t = precision_recall_curve(y_true[:, i], y_score[:, i])
        f1 = 2 * p * r / np.maximum(p + r, 1e-8)
        thresholds.append(float(t[np.argmax(f1[:-1])]))
    return np.array(thresholds)


def compute_metrics(y_true, y_score, thresholds):
    y_pred = (y_score >= thresholds).astype(int)
    rows = []

    for i, name in enumerate(LABELS):
        t = y_true[:, i]
        p = y_pred[:, i]
        tp = int(((p == 1) & (t == 1)).sum())
        fp = int(((p == 1) & (t == 0)).sum())
        fn = int(((p == 0) & (t == 1)).sum())
        tn = int(((p == 0) & (t == 0)).sum())

        both = len(np.unique(t)) == 2
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        rows.append({
            "disease": name,
            "positives": tp + fn,
            "threshold": thresholds[i],
            "auroc": roc_auc_score(t, y_score[:, i]) if both else float("nan"),
            "auprc": average_precision_score(t, y_score[:, i]) if both else float("nan"),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "accuracy": (tp + tn) / len(t),
        })

    return rows


def summarize(name, rows):
    rare = [r for r in rows if r["disease"] in RARE_LABELS]
    keys = ["auroc", "auprc", "precision", "recall", "f1", "specificity", "accuracy"]
    summary = {"model": name}
    for k in keys:
        summary[f"mean_{k}"] = float(np.nanmean([r[k] for r in rows]))
    summary["rare_auroc"] = float(np.nanmean([r["auroc"] for r in rare]))
    summary["rare_f1"] = float(np.nanmean([r["f1"] for r in rare]))
    return summary


def make_loader(split, max_n, args):
    data = ChestXrayDataset(split)
    if max_n > 0:
        data = Subset(data, range(min(max_n, len(data))))
    return DataLoader(
        data, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = sorted(glob.glob("checkpoints/*.pt"))
    names = [os.path.basename(p).replace("densenet121_", "").replace(".pt", "") for p in paths]
    print("Models:", names, flush=True)

    models = []
    for path in paths:
        model = create_model(pretrained=False)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        models.append(model.to(device).eval())

    y_val, val_scores = predict(models, make_loader("val", args.max_val, args), device)
    print("Validation predictions done.", flush=True)
    y_test, test_scores = predict(models, make_loader("test", args.max_test, args), device)
    print("Test predictions done.", flush=True)

    os.makedirs("results", exist_ok=True)
    summaries = []

    for name, v_score, t_score in zip(names, val_scores, test_scores):
        thresholds = best_thresholds(y_val, v_score)
        rows = compute_metrics(y_test, t_score, thresholds)

        with open(f"results/test_metrics_{name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        summaries.append(summarize(name, rows))

    with open("results/test_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print("Test set results:")
    for s in summaries:
        print(
            f"{s['model']}: AUROC={s['mean_auroc']:.4f}, AUPRC={s['mean_auprc']:.4f}, "
            f"precision={s['mean_precision']:.4f}, recall={s['mean_recall']:.4f}, "
            f"F1={s['mean_f1']:.4f}, accuracy={s['mean_accuracy']:.4f}, "
            f"rare AUROC={s['rare_auroc']:.4f}, rare F1={s['rare_f1']:.4f}",
            flush=True
        )


if __name__ == "__main__":
    main()
