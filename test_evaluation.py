
# ==================================================
# KNEE OSTEOARTHRITIS - FINAL TEST EVALUATION
# ==================================================

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import KneeDataset
from model import create_model
from evaluate import calculate_metrics
from args import DEVICE, BATCH_SIZE


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

test_csv = Path("outputs/splits/test.csv")

model_path = Path("out/pretrained/best_model_fold_3.pth")

output_folder = Path("outputs/pretrained")
output_folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load test dataset
# --------------------------------------------------

test_dataset = KneeDataset(test_csv)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("----- Final Test Evaluation -----")
print("Test images:", len(test_dataset))
print("Device:", DEVICE)


# --------------------------------------------------
# 3. Load saved pretrained model
# --------------------------------------------------

model = create_model(num_classes=5)

state_dict = torch.load(
    model_path,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

print("Loaded model:", model_path)


# --------------------------------------------------
# 4. Make predictions on test images
# --------------------------------------------------

all_labels = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        all_labels.extend(labels.numpy())

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

all_labels = np.asarray(all_labels)
all_probabilities = np.asarray(all_probabilities)

predictions = np.argmax(all_probabilities, axis=1)


# --------------------------------------------------
# 5. Calculate classification metrics
# --------------------------------------------------

results = calculate_metrics(
    all_labels,
    all_probabilities,
    num_classes=5
)

print("\n----- Test Results -----")

for metric_name, value in results.items():
    print(f"{metric_name}: {value:.4f}")

print("\nTest class distribution:")
for grade in range(5):
    print(f"KL {grade}: {(all_labels == grade).sum()} images")


# --------------------------------------------------
# 6. Save test metrics
# --------------------------------------------------

results_df = pd.DataFrame([results])

results_path = output_folder / "test_results.csv"

results_df.to_csv(results_path, index=False)

print("\nTest results saved:", results_path)


# --------------------------------------------------
# 7. Generate confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    predictions,
    labels=[0, 1, 2, 3, 4]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[0, 1, 2, 3, 4]
)

fig, ax = plt.subplots(figsize=(7, 6))

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("Pretrained ResNet18 - Test Confusion Matrix")
ax.set_xlabel("Predicted KL Grade")
ax.set_ylabel("Actual KL Grade")

plt.tight_layout()

matrix_path = output_folder / "test_confusion_matrix.png"

plt.savefig(matrix_path, dpi=300)
plt.close()

print("Confusion matrix saved:", matrix_path)

print("\nFinal test evaluation completed!")
