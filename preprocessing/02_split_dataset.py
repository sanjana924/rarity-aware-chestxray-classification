
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# File locations
folder = Path("raw_data")
output = Path("manifests")

df = pd.read_csv(output / "cleaned_metadata.csv")

# Read official NIH split lists
train_names = set(
    (folder / "train_val_list.txt").read_text().splitlines()
)
test_names = set(
    (folder / "test_list.txt").read_text().splitlines()
)

# Verify official splits
assert not (train_names & test_names), "Overlapping images!"
assert len(train_names) == 86524
assert len(test_names) == 25596

train_val = df[df["Image Index"].isin(train_names)].copy()
test = df[df["Image Index"].isin(test_names)].copy()

assert len(train_val) == 86524
assert len(test) == 25596

# Check patient separation in official split
assert set(train_val["Patient ID"]).isdisjoint(
    set(test["Patient ID"])
), "Patient overlap between training and testing!"

# Split by patient ID
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.10,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        train_val,
        groups=train_val["Patient ID"]
    )
)

train = train_val.iloc[train_idx].copy()
val = train_val.iloc[val_idx].copy()

# Verify no patient overlap
assert set(train["Patient ID"]).isdisjoint(
    set(val["Patient ID"])
)

# Save files
train.to_csv(output / "train.csv", index=False)
val.to_csv(output / "val.csv", index=False)
test.to_csv(output / "test.csv", index=False)

print("Training images:", len(train))
print("Validation images:", len(val))
print("Testing images:", len(test))
print("Total:", len(train) + len(val) + len(test))
print("Patient overlap: NONE")
print("Saved train.csv, val.csv, test.csv")
