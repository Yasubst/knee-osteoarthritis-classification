
from pathlib import Path
import pandas as pd

# Best validation Balanced Accuracy from each fold
results = {
    "Fold": [1, 2, 3, 4, 5],
    "Best_Epoch": [2, 2, 2, 2, 2],
    "Balanced_Accuracy": [0.2158, 0.2129, 0.2000, 0.2164, 0.2156],
    "ROC_AUC": [0.5982, 0.5733, 0.5895, 0.5893, 0.5681],
    "Average_Precision": [0.2398, 0.2519, 0.2386, 0.2626, 0.2362]
}

df = pd.DataFrame(results)

# Calculate mean and sample standard deviation
summary = df[
    ["Balanced_Accuracy", "ROC_AUC", "Average_Precision"]
].agg(["mean", "std"])

# Save results
output_folder = Path("outputs")
output_folder.mkdir(parents=True, exist_ok=True)

df.to_csv(output_folder / "cross_validation_results.csv", index=False)
summary.to_csv(output_folder / "cross_validation_summary.csv")

print("----- Five-Fold Results -----")
print(df.to_string(index=False))

print("\n----- Mean and Standard Deviation -----")
print(summary)

print("\nCSV files saved successfully!")
