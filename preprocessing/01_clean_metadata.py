
import pandas as pd
from pathlib import Path

# File locations
input_file = Path("raw_data/Data_Entry_2017_v2020.csv")
output_file = Path("manifests/cleaned_metadata.csv")

# Read the original metadata
df = pd.read_csv(input_file)

print("Total images:", len(df))
print("\nMissing values:")
print(df[["Image Index", "Finding Labels", "Patient ID"]].isna().sum())

# The 14 findings we want to predict
labels = [
    "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Effusion", "Emphysema", "Fibrosis",
    "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

# Validate required metadata
required = ["Image Index", "Finding Labels", "Patient ID"]
if df[required].isna().any().any():
    raise ValueError("Missing required metadata. Inspect the CSV.")

if df["Image Index"].duplicated().any():
    raise ValueError("Duplicate image filenames detected.")

# Validate and encode all finding labels
known = set(labels) | {"No Finding"}
observed = set()
for value in df["Finding Labels"]:
    observed.update(value.split("|"))

unknown = observed - known
if unknown:
    raise ValueError(f"Unexpected finding labels: {unknown}")

for label in labels:
    df[label] = df["Finding Labels"].apply(
        lambda value: int(label in value.split("|"))
    )

# Keep relevant columns
cleaned = df[
    ["Image Index", "Patient ID", "Finding Labels"] + labels
].copy()

output_file.parent.mkdir(parents=True, exist_ok=True)
cleaned.to_csv(output_file, index=False)

print("\nCleaned dataset shape:", cleaned.shape)
print("Missing values after cleaning:", cleaned.isna().sum().sum())
print("Saved:", output_file)
