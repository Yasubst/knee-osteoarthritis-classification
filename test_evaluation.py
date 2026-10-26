
# ==================================================
# KNEE OSTEOARTHRITIS - FINAL TEST EVALUATION
# ==================================================

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    classification_report,
)

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
# 5. Calculate Precision and Recall
#    Micro, Macro, and Weighted Averaging
# --------------------------------------------------

print("\n----- Precision and Recall Results -----")

averaging_results = {}

for average_type in ["micro", "macro", "weighted"]:

    precision, recall, _, _ = precision_recall_fscore_support(
        all_labels,
        predictions,
        labels=[0, 1, 2, 3, 4],
        average=average_type,
        zero_division=0
    )

    averaging_results[f"{average_type}_precision"] = precision
    averaging_results[f"{average_type}_recall"] = recall

    print(f"\n{average_type.upper()} AVERAGING")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")


# --------------------------------------------------
# 6. Classification report by KL grade
# --------------------------------------------------

print("\n----- Precision and Recall by KL Grade -----")

report_text = classification_report(
    all_labels,
    predictions,
    labels=[0, 1, 2, 3, 4],
    target_names=["KL0", "KL1", "KL2", "KL3", "KL4"],
    digits=4,
    zero_division=0
)

print(report_text)

# Save detailed classification report

report_dict = classification_report(
    all_labels,
    predictions,
    labels=[0, 1, 2, 3, 4],
    target_names=["KL0", "KL1", "KL2", "KL3", "KL4"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()

report_path = output_folder / "test_classification_report.csv"

report_df.to_csv(report_path, index=True)

print("Classification report saved:", report_path)


# --------------------------------------------------
# 7. Calculate existing classification metrics
# --------------------------------------------------

results = calculate_metrics(
    all_labels,
    all_probabilities,
    num_classes=5
)

print("\n----- Test Results -----")

for metric_name, value in results.items():
    print(f"{metric_name}: {value:.4f}")


# --------------------------------------------------
# 8. Display test class distribution
# --------------------------------------------------

print("\nTest class distribution:")

for grade in range(5):
    print(f"KL {grade}: {(all_labels == grade).sum()} images")


# --------------------------------------------------
# 9. Save all test metrics
# --------------------------------------------------

# Combine existing metrics with precision and recall
results.update(averaging_results)

results_df = pd.DataFrame([results])

results_path = output_folder / "test_results.csv"

results_df.to_csv(results_path, index=False)

print("\nTest results saved:", results_path)


# --------------------------------------------------
# 10. Generate confusion matrix
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
