
# ==================================================
# KNEE OSTEOARTHRITIS CLASSIFICATION
# CLASSIFICATION METRICS
# ==================================================

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score
)


# ==================================================
# CALCULATE CLASSIFICATION METRICS
# ==================================================

def calculate_metrics(labels, probabilities, num_classes=5):

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    # Predicted class = highest probability
    predictions = np.argmax(probabilities, axis=1)

    # Ordinary accuracy
    accuracy = accuracy_score(labels, predictions)

    # Balanced Accuracy
    balanced_accuracy = balanced_accuracy_score(
        labels,
        predictions
    )

    # Convert labels into one-vs-rest binary format
    binary_labels = (
        labels[:, None] ==
        np.arange(num_classes)[None, :]
    ).astype(int)

    auc_scores = []
    ap_scores = []

    # Calculate metrics for each KL grade
    for grade in range(num_classes):

        true_binary = binary_labels[:, grade]
        predicted_scores = probabilities[:, grade]

        # ROC-AUC requires both positive and negative examples
        if len(np.unique(true_binary)) == 2:

            auc_scores.append(
                roc_auc_score(true_binary, predicted_scores)
            )

            ap_scores.append(
                average_precision_score(true_binary, predicted_scores)
            )

    # Macro averages
    roc_auc = (
        float(np.mean(auc_scores))
        if auc_scores else float("nan")
    )

    average_precision = (
        float(np.mean(ap_scores))
        if ap_scores else float("nan")
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "average_precision": average_precision
    }


# ==================================================
# TEST METRIC CALCULATIONS
# ==================================================

if __name__ == "__main__":

    # Five sample images, one from each KL grade
    sample_labels = [0, 1, 2, 3, 4]

    # Example predicted probabilities
    sample_probabilities = [
        [0.70, 0.10, 0.10, 0.05, 0.05],
        [0.10, 0.60, 0.10, 0.10, 0.10],
        [0.10, 0.10, 0.60, 0.10, 0.10],
        [0.05, 0.10, 0.10, 0.65, 0.10],
        [0.05, 0.05, 0.10, 0.10, 0.70]
    ]

    results = calculate_metrics(
        sample_labels,
        sample_probabilities
    )

    print("----- Metric Calculation Test -----")

    for metric_name, value in results.items():
        print(metric_name, ":", round(value, 4))
