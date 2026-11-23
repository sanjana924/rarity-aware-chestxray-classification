
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DATA_ROOT = "/pscratch/sd/p/pravch1/chestxray_project"

LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Effusion", "Emphysema", "Fibrosis",
    "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]


class ChestXrayDataset(Dataset):
    def __init__(self, split="train"):
        csv_path = os.path.join(
            DATA_ROOT, "manifests", f"{split}.csv"
        )
        self.df = pd.read_csv(csv_path)

        self.image_dir = os.path.join(
            DATA_ROOT, "raw_data", "images"
        )

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = os.path.join(
            self.image_dir, row["Image Index"]
        )

        with Image.open(image_path) as img:
            image = self.transform(img)

        labels = torch.tensor(
            row[LABELS].to_numpy(dtype="float32"),
            dtype=torch.float32
        )

        return image, labels


if __name__ == "__main__":
    dataset = ChestXrayDataset("train")

    print("Training images:", len(dataset))

    image, labels = dataset[0]

    print("Image shape:", image.shape)
    print("Label shape:", labels.shape)
    print("Labels:", labels)
