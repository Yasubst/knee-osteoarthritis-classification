
# ==================================================
#    TEST PYTORCH DATALOADER
# ==================================================

from torch.utils.data import DataLoader
from dataset import KneeDataset


# 1. Load Fold 1 training dataset
train_dataset = KneeDataset(
    "outputs/splits/train_fold_1.csv"
)


# 2. Create a DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)


# 3. Display dataset information
print("----- DataLoader Information -----")
print("Total images:", len(train_dataset))
print("Batch size:", train_loader.batch_size)
print("Total batches:", len(train_loader))


# 4. Load one batch
images, labels = next(iter(train_loader))


# 5. Display batch information
print("\n----- First Batch -----")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Image data type:", images.dtype)
print("Labels:", labels.tolist())
