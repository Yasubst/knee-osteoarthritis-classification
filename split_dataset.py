#==================================================
#       PREPARE IMAGE PATHS AND LABELS
#==================================================
from pathlib import Path
from collections import Counter

# Location of labeled knee X-ray images
LABELED_FOLDER = Path("data/OSAIL_KL_Dataset/Labeled")

# Empty lists to store image paths and KL grades
image_paths = []
labels = []

# Loop through KL grades 0 to 4
for grade in range(5):
    grade_folder = LABELED_FOLDER / str(grade)

    # Find all PNG images in the current grade folder
    for image_path in sorted(grade_folder.glob("*.png")):

        # store the image path
        image_paths.append(str(image_path))

        # Store the corresponding KL grade
        labels.append(grade)

# Display the results
print("----- Dataset Preparation -----")
print("Total image paths:", len(image_paths))
print("Total labels:", len(labels))

print("\n----- KL Grade Distribution -----")
grade_counts = Counter(labels)

for grade in range(5):
    print(f"KL Grade {grade}: {grade_counts[grade]} images")

print("\nFirst image path:", image_paths[0])
print("First image label:", labels[0])


# ==================================================
#      CHECK FILENAMES AND METADATA
# ==================================================

# Display three sample filenames from each KL grade
print("\n----- Sample Filenames -----")

for grade in range(5):
    grade_folder = LABELED_FOLDER / str(grade)

    sample_files = sorted(grade_folder.glob("*.png"))[:3]

    print(f"\nKL Grade {grade}:")

    for file in sample_files:
        print(" ", file.name)


# Check for metadata files in the dataset folder
dataset_folder = LABELED_FOLDER.parent

print("\n----- Dataset Metadata Files -----")

metadata_extensions = {".csv", ".xlsx", ".json", ".txt"}

metadata_files = [
    file
    for file in dataset_folder.rglob("*")
    if file.is_file()
    and file.suffix.lower() in metadata_extensions
]

if metadata_files:
    for file in metadata_files:
        print(file.relative_to(dataset_folder))
else:
    print("No CSV, Excel, JSON, or TXT metadata files found.")


#===============================================
#              STRATIFIED TRAIN / TEST SPLIT
#===============================================

from sklearn.model_selection import train_test_split

# Split image paths and labels into two datasets
development_paths, test_paths, development_labels, test_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

# Display dataset sizes
print("\n----- Dataset Split Results -----")
print("Total images:", len(image_paths))
print("Development images:", len(development_paths))
print("Test images:", len(test_paths))

# Check KL grade distribution
print("\n----- Development Set Distribution -----")
development_counts = Counter(development_labels)

for grade in range(5):
    print(f"KL Grade {grade}: {development_counts[grade]} images")

print("\n----- Test Set Distribution -----")
test_counts = Counter(test_labels)

for grade in range(5):
    print(f"KL Grade {grade}: {test_counts[grade]} images")

# Verify that no image path appears in both sets
overlap = set(development_paths) & set(test_paths)

print("\n----- Data Leakage Check -----")
print("Overlapping image paths:", len(overlap))

#======================================================
#          STRATIFIED 5-FOLD CROSS-VALIDATION
#=========================================================

from sklearn.model_selection import StratifiedKFold

# create the 5-fold splitter
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n---------- 5-Fold Cross-Validation-------------")

# Generate training and validation indices
for fold, (train_idx, val_idx) in enumerate(
    skf.split(development_paths, development_labels),
    start=1
):

    # Select image paths for this fold
    train_paths = [development_paths[i] for i in train_idx]
    val_paths = [development_paths[i] for i in val_idx]

    # Select labels for this fold
    train_labels = [development_labels[i] for i in train_idx]
    val_labels = [development_labels[i] for i in val_idx]

    # Display fold information
    print(f"\nFold {fold}")
    print("Training images:", len(train_paths))
    print("Validation images:", len(val_paths))

    # Check whether image paths overlap
    overlap = set(train_paths) & set(val_paths)

    print("Overlapping paths:", len(overlap))

    # Display validation class distribution
    val_counts = Counter(val_labels)

    for grade in range(5):
        print(f"  KL Grade {grade}: {val_counts[grade]}")



# ==================================================
#       SAVE DATASET SPLITS TO CSV
# ==================================================

import csv

# Create a folder to store split information
split_folder = Path("outputs/splits")
split_folder.mkdir(parents=True, exist_ok=True)


# Function to save image paths and labels to a CSV file
def save_split_csv(file_path, paths, labels):

    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.writer(csv_file)

        # Write column names
        writer.writerow(["image_path", "label"])

        # Write each image path and its corresponding label
        for image_path, label in zip(paths, labels):
            writer.writerow([image_path, label])


# Save the development and test datasets
save_split_csv(
    split_folder / "development.csv",
    development_paths,
    development_labels
)

save_split_csv(
    split_folder / "test.csv",
    test_paths,
    test_labels
)


# Save training and validation datasets for each fold
for fold, (train_idx, val_idx) in enumerate(
    skf.split(development_paths, development_labels),
    start=1
):

    train_paths = [development_paths[i] for i in train_idx]
    val_paths = [development_paths[i] for i in val_idx]

    train_labels = [development_labels[i] for i in train_idx]
    val_labels = [development_labels[i] for i in val_idx]

    save_split_csv(
        split_folder / f"train_fold_{fold}.csv",
        train_paths,
        train_labels
    )

    save_split_csv(
        split_folder / f"val_fold_{fold}.csv",
        val_paths,
        val_labels
    )


print("\n----- CSV Export Completed -----")
print("CSV files saved to:", split_folder.resolve())

for file in sorted(split_folder.glob("*.csv")):
    print(file.name)
