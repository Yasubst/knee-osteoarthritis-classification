
# ==================================================
# KNEE OSTEOARTHRITIS CLASSIFICATION
# FIVE-FOLD CROSS-VALIDATION TRAINING
# ==================================================

import csv
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import KneeDataset

from trainer import (
    initialize_training,
    train_one_epoch,
    validate_one_epoch,
    save_best_model
)

from args import (
    DEVICE,
    EPOCHS,
    NUM_FOLDS,
    BATCH_SIZE
)

from utils import save_fold_plots


# ==================================================
# 1. CREATE DATA LOADERS FOR EACH FOLD
# ==================================================

def create_fold_dataloaders(fold):

    split_folder = Path("outputs/splits")

    # Load training dataset
    train_dataset = KneeDataset(
        split_folder / f"train_fold_{fold}.csv"
    )

    # Load validation dataset
    val_dataset = KneeDataset(
        split_folder / f"val_fold_{fold}.csv"
    )

    # Training DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # Validation DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


# ==================================================
# 2. MAIN TRAINING PROCESS
# ==================================================

if __name__ == "__main__":

    print("----- Five-Fold Pretrained CNN Training -----")
    print("Device:", DEVICE)
    print("Epochs:", EPOCHS)

    folds_to_run = NUM_FOLDS
    # Store best results from each fold
    fold_results = []

    # Train each fold separately
    for fold in range(1, folds_to_run + 1):

        print(f"\n========== FOLD {fold} ==========")

        # Load datasets for this fold
        train_loader, val_loader = create_fold_dataloaders(
            fold
        )

        print("Training images:", len(train_loader.dataset))
        print("Validation images:", len(val_loader.dataset))

        # Create fresh model and optimizer for this fold
        model, criterion, optimizer = initialize_training()

        # Track best validation Balanced Accuracy
        best_score = -1.0

        # Store the best validation results for this fold
        best_fold_result = None

        # Store training and validation metrics
        history = {
            "train_loss": [],
            "val_loss": [],

            "train_balanced_accuracy": [],
            "val_balanced_accuracy": [],

            "train_roc_auc": [],
            "val_roc_auc": [],

            "train_ap": [],
            "val_ap": []
        }

        # ==========================================
        # 3. TRAIN FOR MULTIPLE EPOCHS
        # ==========================================

        for epoch in range(EPOCHS):

            print(f"\nEpoch {epoch + 1}/{EPOCHS}")

            # Train one epoch
            train_loss, train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                DEVICE
            )

            # Validate one epoch
            (
                val_loss,
                val_accuracy,
                val_balanced_accuracy,
                val_roc_auc,
                val_ap
            ) = validate_one_epoch(
                model,
                val_loader,
                criterion,
                DEVICE
            )

            # Check whether this epoch achieved a new best score
            is_best = val_balanced_accuracy > best_score

            # Save the best model
            best_score = save_best_model(
                model=model,
                fold=fold,
                balanced_accuracy=val_balanced_accuracy,
                best_score=best_score
            )

            # Record metrics from the best epoch
            if is_best:
                best_fold_result = {
                    "fold": fold,
                    "best_epoch": epoch + 1,
                    "balanced_accuracy": val_balanced_accuracy,
                    "roc_auc": val_roc_auc,
                    "average_precision": val_ap
                }

            # ======================================
            # 4. RECORD METRICS
            # ======================================

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            history["train_balanced_accuracy"].append(
                train_metrics["balanced_accuracy"]
            )

            history["val_balanced_accuracy"].append(
                val_balanced_accuracy
            )

            history["train_roc_auc"].append(
                train_metrics["roc_auc"]
            )

            history["val_roc_auc"].append(
                val_roc_auc
            )

            history["train_ap"].append(
                train_metrics["average_precision"]
            )

            history["val_ap"].append(val_ap)

            # ======================================
            # 5. PRINT EPOCH RESULTS
            # ======================================

            print("Training loss:", round(train_loss, 4))
            print("Validation loss:", round(val_loss, 4))

            print(
                "Training Balanced Accuracy:",
                round(
                    train_metrics["balanced_accuracy"] * 100,
                    2
                ),
                "%"
            )

            print(
                "Validation Balanced Accuracy:",
                round(val_balanced_accuracy * 100, 2),
                "%"
            )

            print(
                "Validation ROC-AUC:",
                round(val_roc_auc, 4)
            )

            print(
                "Validation Average Precision:",
                round(val_ap, 4)
            )

        # ==========================================
        # 6. FINISH FOLD
        # ==========================================

        print(
            f"\nFold {fold} completed. "
            f"Best Balanced Accuracy: {best_score * 100:.2f}%"
        )

        # Save training and validation graphs
        save_fold_plots(history, fold)

        # Store the best result from this fold
        if best_fold_result is not None:
            fold_results.append(best_fold_result)

            # Save results after each completed fold
            output_folder = Path("outputs/pretrained")
            output_folder.mkdir(parents=True, exist_ok=True)

            csv_path = output_folder / "cross_validation_results.csv"

            with open(csv_path, "w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "fold",
                        "best_epoch",
                        "balanced_accuracy",
                        "roc_auc",
                        "average_precision"
                    ]
                )

                writer.writeheader()
                writer.writerows(fold_results)

            print("Progress saved:", csv_path)




    print("\nFive-fold training completed successfully!")
