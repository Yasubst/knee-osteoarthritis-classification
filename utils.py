
from pathlib import Path
import matplotlib.pyplot as plt


def save_fold_plots(history, fold):

    plot_folder = Path("outputs/pretrained")
    plot_folder.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Four required plots
    plots = {
        "loss": (
            "train_loss",
            "val_loss",
            "Loss"
        ),
        "balanced_accuracy": (
            "train_balanced_accuracy",
            "val_balanced_accuracy",
            "Balanced Accuracy"
        ),
        "roc_auc": (
            "train_roc_auc",
            "val_roc_auc",
            "ROC-AUC"
        ),
        "average_precision": (
            "train_ap",
            "val_ap",
            "Average Precision"
        )
    }

    for name, (train_key, val_key, title) in plots.items():

        plt.figure(figsize=(8, 5))

        plt.plot(
            epochs,
            history[train_key],
            marker="o",
            label="Training"
        )

        plt.plot(
            epochs,
            history[val_key],
            marker="s",
            label="Validation"
        )

        plt.title(f"Fold {fold} - {title}")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.xticks(list(epochs))

        if name != "loss":
            plt.ylim(0, 1)

        plt.legend()
        plt.grid(alpha=0.3)

        output_path = plot_folder / f"fold_{fold}_{name}.png"

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print("Plot saved:", output_path)
if __name__ == "__main__":

    test_history = {
        "train_loss": [1.5, 1.4],
        "val_loss": [1.6, 1.5],
        "train_balanced_accuracy": [0.20, 0.25],
        "val_balanced_accuracy": [0.21, 0.23],
        "train_roc_auc": [0.50, 0.55],
        "val_roc_auc": [0.52, 0.54],
        "train_ap": [0.22, 0.27],
        "val_ap": [0.23, 0.25]
    }

    save_fold_plots(test_history, fold=1)