
import csv
from pathlib import Path
from statistics import mean, stdev

results_dir = Path(__file__).parent / "results"

configs = {
    "Default": "default",
    "RPB4": "rarity_rpb4",
    "RPB8": "rarity",
    "RPB16": "rarity_rpb16",
}

total_gpu_hours = 0.0

for name, tag in configs.items():
    best_aurocs = []
    gpu_hours = []

    for seed in (0, 1, 2):
        path = results_dir / f"history_1gpu_{tag}_seed{seed}.csv"

        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))

        best_aurocs.append(max(float(r["val_auroc"]) for r in rows))
        gpu_hours.append(sum(float(r["gpu_hours"]) for r in rows))

    total_gpu_hours += sum(gpu_hours)

    print(f"\n{name}")
    print(f"  Best AUROC by seed: {[round(x, 4) for x in best_aurocs]}")
    print(f"  Mean best AUROC: {mean(best_aurocs):.4f}")
    print(f"  AUROC standard deviation: {stdev(best_aurocs):.4f}")
    print(f"  Training GPU-hours: {sum(gpu_hours):.4f}")

print(f"\nTOTAL TRAINING GPU-HOURS: {total_gpu_hours:.4f}")
