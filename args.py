
# ==================================================
#      MODEL AND TRAINING CONFIGURATION
# ==================================================

from pathlib import Path
import torch


# ---------------- MODEL SETTINGS ----------------

MODEL_NAME = "resnet18"

NUM_CLASSES = 5

# False means training from scratch
PRETRAINED = True


# ---------------- TRAINING SETTINGS ----------------

BATCH_SIZE = 16

LEARNING_RATE = 0.0001

# Train pretrained ResNet18 for 5 epochs
EPOCHS = 5

NUM_FOLDS = 5

RANDOM_SEED = 42


# ---------------- DEVICE SETTINGS ----------------

# Use GPU if available, otherwise use CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------- FOLDER PATHS ----------------

SPLIT_FOLDER = Path("outputs/splits")

MODEL_FOLDER = Path("out/pretrained")

PLOT_FOLDER = Path("outputs/pretrained")

# Create folders if they do not already exist
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
PLOT_FOLDER.mkdir(parents=True, exist_ok=True)


# ---------------- TEST CONFIGURATION ----------------

if __name__ == "__main__":

    print("----- Training Configuration -----")
    print("Model:", MODEL_NAME)
    print("Number of classes:", NUM_CLASSES)
    print("Pretrained:", PRETRAINED)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Epochs:", EPOCHS)
    print("Number of folds:", NUM_FOLDS)
    print("Random seed:", RANDOM_SEED)
    print("Device:", DEVICE)
    print("Model folder:", MODEL_FOLDER)
    print("Plot folder:", PLOT_FOLDER)
