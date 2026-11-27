import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler

from dataset import DATA_ROOT
from rarity_sampler import RarityAwareSampler, RARE_LABELS

GPUS = 2
BATCH = 16

df = pd.read_csv(os.path.join(DATA_ROOT, "manifests", "train.csv"))
rare = df[RARE_LABELS].to_numpy().any(axis=1)

print("Rare labels:", RARE_LABELS)
for name in RARE_LABELS:
    print(f"  {name}: {int(df[name].sum())} train images")
print(f"Images with any rare label: {rare.sum()} of {len(df)} ({rare.mean()*100:.1f}%)")

results = {}
for name in ["default", "rarity"]:
    per_gpu = []
    for r in range(GPUS):
        if name == "default":
            s = DistributedSampler(range(len(df)), num_replicas=GPUS, rank=r, shuffle=True, seed=42)
        else:
            s = RarityAwareSampler(df, BATCH, 2, num_replicas=GPUS, rank=r, seed=42)
        s.set_epoch(0)
        per_gpu.append(list(s))
    steps = min(len(p) for p in per_gpu) // BATCH
    counts = np.array([
        sum(rare[p[i * BATCH:(i + 1) * BATCH]].sum() for p in per_gpu)
        for i in range(steps)
    ])
    results[name] = counts
    print(f"{name}: avg rare images per step = {counts.mean():.2f}, "
          f"steps with zero rare images = {(counts == 0).mean()*100:.1f}%")

os.makedirs("results", exist_ok=True)
pd.DataFrame(results).to_csv("results/rare_per_step_2gpu.csv", index_label="step")

fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.arange(0, max(c.max() for c in results.values()) + 2) - 0.5
ax.hist(results["default"], bins=bins, alpha=0.7, label="Default sampler")
ax.hist(results["rarity"], bins=bins, alpha=0.7, label="Rarity-aware sampler")
ax.set_xlabel(f"Rare-label images per training step ({GPUS} GPUs x {BATCH} images)")
ax.set_ylabel("Number of steps")
ax.legend()
fig.tight_layout()
fig.savefig("results/rare_per_step_2gpu.png", dpi=150)
print("Saved results/rare_per_step_2gpu.csv and results/rare_per_step_2gpu.png")
