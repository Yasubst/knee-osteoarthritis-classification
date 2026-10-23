
# ==================================================
# KNEE OSTEOARTHRITIS CLASSIFICATION
# CNN TRAINER - TRAINING AND VALIDATION TEST
# ==================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


from pathlib import Path
from torch.utils.data import DataLoader, Subset
from evaluate import calculate_metrics

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score
)

from model import create_model
from dataset import KneeDataset

from args import (
    NUM_CLASSES,
    LEARNING_RATE,
    DEVICE,
    EPOCHS
)


# ==================================================
# CREATE A FRESH MODEL FOR EACH FOLD
# ==================================================

def initialize_training():

    # Create a new ResNet18 model
    model = create_model(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    # Define classification loss
    criterion = nn.CrossEntropyLoss()

    # Create a new Adam optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    return model, criterion, optimizer



# ==================================================
# TRAIN ONE EPOCH AND CALCULATE METRICS
# ==================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):

    model.train()

    total_loss = 0.0

    all_labels = []
    all_probabilities = []

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Convert predictions to probabilities
        probabilities = torch.softmax(outputs.detach(), dim=1)

        # Store results
        all_labels.extend(labels.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

        # Accumulate loss
        total_loss += loss.item() * images.size(0)

    # Average training loss
    average_loss = total_loss / len(train_loader.dataset)

    # Calculate training metrics
    metrics = calculate_metrics(
        all_labels,
        all_probabilities,
        num_classes=NUM_CLASSES
    )

    return average_loss, metrics


# ==================================================
# 2. VALIDATE THE MODEL
# ==================================================

def validate_one_epoch(model, val_loader, criterion, device):

    # Set model to evaluation mode
    model.eval()

    total_loss = 0.0
    total_images = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    # Disable gradient calculations
    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Convert scores to probabilities
            probabilities = torch.softmax(outputs, dim=1)

            # Calculate loss
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_images += labels.size(0)

            # Predicted KL grades
            predictions = torch.argmax(outputs, dim=1)

            # Store results
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(
                probabilities.cpu().tolist()
            )

    # Average validation loss
    average_loss = total_loss / total_images

    # Ordinary accuracy
    correct = sum(
        true == predicted
        for true, predicted in zip(
            all_labels, all_predictions
        )
    )

    accuracy = correct / total_images

    # Balanced Accuracy
    balanced_accuracy = balanced_accuracy_score(
        all_labels,
        all_predictions
    )

    # Convert results into NumPy arrays
    labels_array = np.array(all_labels)
    probabilities_array = np.array(all_probabilities)

    # One-vs-rest binary labels
    binary_labels = (
        labels_array[:, None] ==
        np.arange(NUM_CLASSES)[None, :]
    ).astype(int)

    roc_auc_scores = []
    ap_scores = []

    # Calculate ROC-AUC and AP for each KL grade
    for grade in range(NUM_CLASSES):

        true_binary = binary_labels[:, grade]

        predicted_probabilities = (
            probabilities_array[:, grade]
        )

        # Both positive and negative samples are needed
        if len(np.unique(true_binary)) == 2:

            auc = roc_auc_score(
                true_binary,
                predicted_probabilities
            )

            ap = average_precision_score(
                true_binary,
                predicted_probabilities
            )

            roc_auc_scores.append(auc)
            ap_scores.append(ap)

    # Macro-average ROC-AUC
    roc_auc = (
        float(np.mean(roc_auc_scores))
        if roc_auc_scores
        else float("nan")
    )

    # Macro-average Average Precision
    average_precision = (
        float(np.mean(ap_scores))
        if ap_scores
        else float("nan")
    )

    return (
        average_loss,
        accuracy,
        balanced_accuracy,
        roc_auc,
        average_precision
    )


# ==================================================
# 3. SAVE THE BEST MODEL
# ==================================================

def save_best_model(model, fold, balanced_accuracy, best_score):

    # Save only if Balanced Accuracy improves
    if balanced_accuracy > best_score:

        output_folder = Path("out/pretrained")
        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = (
            output_folder /
            f"best_model_fold_{fold}.pth"
        )

        # Save model weights
        torch.save(
            model.state_dict(),
            model_path
        )

        print("New best model saved:", model_path)

        best_score = balanced_accuracy

    return best_score


# ==================================================
# 4. MAIN - TEST TWO TRAINING EPOCHS
# ==================================================

if __name__ == "__main__":

    # Create a fresh CNN model
    model, criterion, optimizer = initialize_training()
    # Load Fold 1 datasets
    train_dataset = KneeDataset(
        "outputs/splits/train_fold_1.csv"
    )

    val_dataset = KneeDataset(
        "outputs/splits/val_fold_1.csv"
    )

    # Small subsets for CPU testing
    small_train = Subset(
        train_dataset,
        range(32)
    )

    small_val = Subset(
        val_dataset,
        range(16)
    )

    # Training DataLoader
    train_loader = DataLoader(
        small_train,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    # Validation DataLoader
    val_loader = DataLoader(
        small_val,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    print("----- Training and Validation Test -----")
    print("Device:", DEVICE)
    print("Training images:", len(small_train))
    print("Validation images:", len(small_val))

    # Track the best Balanced Accuracy
    best_score = -1.0

    # Store results for plotting later
    history = {
        "train_loss": [],
        "val_loss": [],

        "train_balanced_accuracy": [],
        "val_balanced_accuracy": [],

        "train_roc_auc": [],
        "val_roc_auc": [],

        "train_ap": [],
        "val_ap": [],

        "val_accuracy": []
    }

    # ==========================================
    # 5. TRAIN FOR MULTIPLE EPOCHS
    # ==========================================

    for epoch in range(EPOCHS):

        print(
            f"\n----- Epoch {epoch + 1}/{EPOCHS} -----"
        )

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

        # Record training history

        # Loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Balanced Accuracy
        history["train_balanced_accuracy"].append(
            train_metrics["balanced_accuracy"]
        )
        history["val_balanced_accuracy"].append(
            val_balanced_accuracy
        )

        # ROC-AUC
        history["train_roc_auc"].append(
            train_metrics["roc_auc"]
        )
        history["val_roc_auc"].append(val_roc_auc)

        # Average Precision
        history["train_ap"].append(
            train_metrics["average_precision"]
        )
        history["val_ap"].append(val_ap)

        # Ordinary validation accuracy
        history["val_accuracy"].append(val_accuracy)


        # Save best model
        best_score = save_best_model(
            model=model,
            fold=1,
            balanced_accuracy=val_balanced_accuracy,
            best_score=best_score
        )

        # Print results
        print("Training loss:", round(train_loss, 4))
        print("Validation loss:", round(val_loss, 4))
        print("Training Balanced Accuracy:",
              round(train_metrics["balanced_accuracy"] * 100, 2), "%")

        print("Training ROC-AUC:",
              round(train_metrics["roc_auc"], 4))

        print("Training Average Precision:",
              round(train_metrics["average_precision"], 4))
        print(
            "Validation accuracy:",
            round(val_accuracy * 100, 2),
            "%"
        )
        print(
            "Balanced Accuracy:",
            round(val_balanced_accuracy * 100, 2),
            "%"
        )
        print("ROC-AUC:", round(val_roc_auc, 4))
        print("Average Precision:", round(val_ap, 4))

    # ==========================================
    # 6. FINAL SUMMARY
    # ==========================================

    print("\n----- Training Completed -----")
    print(
        "Best Balanced Accuracy:",
        round(best_score * 100, 2),
        "%"
    )
    print(
        "Recorded epochs:",
        len(history["train_loss"])
    )


    # ==========================================
    # 7. SAVE TRAINING LOSS PLOT
    # ==========================================

    # Create the plots folder
    plot_folder = Path("outputs/plots")
    plot_folder.mkdir(parents=True, exist_ok=True)

    # Epoch numbers
    epochs = range(1, len(history["train_loss"]) + 1)

    # Create a new figure
    plt.figure(figsize=(8, 5))

    # Plot training loss
    plt.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Training Loss"
    )

    # Plot validation loss
    plt.plot(
        epochs,
        history["val_loss"],
        marker="o",
        label="Validation Loss"
    )

    plt.title("Fold 1 - Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(list(epochs))
    plt.legend()
    plt.grid(alpha=0.3)

    # Save the plot
    plot_path = plot_folder / "fold_1_loss_test.png"

    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Loss plot saved:", plot_path)



    # ==========================================
    # 8. SAVE TRAINING AND VALIDATION METRIC PLOTS
    # ==========================================

    metric_plots = {
        "balanced_accuracy": {
            "train": history["train_balanced_accuracy"],
            "val": history["val_balanced_accuracy"],
            "title": "Balanced Accuracy",
            "ylabel": "Balanced Accuracy"
        },

        "roc_auc": {
            "train": history["train_roc_auc"],
            "val": history["val_roc_auc"],
            "title": "ROC-AUC",
            "ylabel": "ROC-AUC"
        },

        "average_precision": {
            "train": history["train_ap"],
            "val": history["val_ap"],
            "title": "Average Precision",
            "ylabel": "Average Precision"
        }
    }

    for metric_name, metric_info in metric_plots.items():

        plt.figure(figsize=(8, 5))

        # Training line
        plt.plot(
            epochs,
            metric_info["train"],
            marker="o",
            label="Training"
        )

        # Validation line
        plt.plot(
            epochs,
            metric_info["val"],
            marker="s",
            label="Validation"
        )

        plt.title(f"Fold 1 - {metric_info['title']}")
        plt.xlabel("Epoch")
        plt.ylabel(metric_info["ylabel"])
        plt.xticks(list(epochs))
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(alpha=0.3)

        plot_path = (
            plot_folder /
            f"fold_1_{metric_name}_test.png"
        )

        plt.savefig(
            plot_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print("Metric plot saved:", plot_path)
