
import pandas as pd
from pathlib import Path

folder = Path("manifests")

train = pd.read_csv(folder / "train.csv")
val = pd.read_csv(folder / "val.csv")
test = pd.read_csv(folder / "test.csv")

labels = [
    "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Effusion", "Emphysema", "Fibrosis",
    "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

# Count positive examples for every finding
results = pd.DataFrame({
    "Finding": labels,
    "Train_Count": train[labels].sum().values,
    "Validation_Count": val[labels].sum().values,
    "Test_Count": test[labels].sum().values
})

# Calculate training prevalence
results["Train_Percentage"] = (
    results["Train_Count"] / len(train) * 100
).round(3)

# Sort from rarest to most common
results = results.sort_values("Train_Count")

# Save the table
results.to_csv(folder / "label_frequencies.csv", index=False)

print(results.to_string(index=False))
print("\nSaved manifests/label_frequencies.csv")

# Warn if rare findings have no positive examples
for _, row in results.iterrows():
    if row["Validation_Count"] == 0:
        print("WARNING: No validation examples for", row["Finding"])
    if row["Test_Count"] == 0:
        print("WARNING: No test examples for", row["Finding"])
