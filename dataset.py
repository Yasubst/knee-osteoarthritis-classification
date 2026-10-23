
# ==================================================
#       PYTORCH DATASET CLASS
# ==================================================

from pathlib import Path

import pandas as pd
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset


class KneeDataset(Dataset):

    def __init__(self, csv_file):

        # Read image paths and labels from CSV
        self.data = pd.read_csv(csv_file)

    def __len__(self):

        # Return total number of images
        return len(self.data)

    def __getitem__(self, index):

        # Get image path from CSV
        image_path = Path(self.data.iloc[index]["image_path"])

        # Get KL grade label
        label = int(self.data.iloc[index]["label"])

        # Open image in grayscale mode
        with Image.open(image_path) as img:
            image = np.array(img.convert("L"))

        # Normalize pixel values to 0–1
        image = image.astype(np.float32) / 255.0

        # Convert NumPy array to PyTorch tensor
        image_tensor = torch.from_numpy(image)

        # Add channel dimension: (224, 224) → (1, 224, 224)
        image_tensor = image_tensor.unsqueeze(0)

        # Convert KL grade into an integer tensor
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor

# ==================================================
# TEST THE DATASET CLASS
# ==================================================

if __name__ == "__main__":

    # Load Fold 1 training CSV
    dataset = KneeDataset("outputs/splits/train_fold_1.csv")

    print("Total training images:", len(dataset))

    # Load the first image and label
    image, label = dataset[0]

    print("Image tensor shape:", image.shape)
    print("Image data type:", image.dtype)
    print("Minimum pixel value:", image.min().item())
    print("Maximum pixel value:", image.max().item())
    print("KL grade label:", label.item())
