
# ==================================================
# PRETRAINED RESNET18 - CROSS-VALIDATION SUMMARY
# ==================================================

import pandas as pd
from pathlib import Path

# Read results from all five folds
input_path = Path(
    "outputs/pretrained/cross_validation_results.csv"
)

df = pd.read_csv(input_path)

# Calculate mean and standard deviation
metrics = [
    "balanced_accuracy",
    "roc_auc",
    "average_precision"
]

summary = df[metrics].agg(["mean", "std"]).T

# Save the summary
output_path = Path(
    "outputs/pretrained/cross_validation_summary.csv"
)

summary.to_csv(output_path)

# Display results
print("----- Pretrained ResNet18 Summary -----")
print(summary.round(4))
print("\nSummary saved:", output_path)
