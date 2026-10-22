
# ==================================================
#        INSPECT THE KNEE X-RAY DATASET
# ==================================================

from pathlib import Path

# Location of our classification dataset
DATASET_FOLDER = Path("data/OSAIL_KL_Dataset")

# Separate labeled and unlabeled data
LABELED_FOLDER = DATASET_FOLDER / "Labeled"
UNLABELED_FOLDER = DATASET_FOLDER / "Unlabeled"

# Check whether the folders exist
print("Labeled folder exists:", LABELED_FOLDER.exists())
print("Unlabeled folder exists:", UNLABELED_FOLDER.exists())

# Show what is inside the labeled folder
print("\nContents of Labeled folder:")

if LABELED_FOLDER.exists():

    for item in sorted(LABELED_FOLDER.iterdir()):
        print(item.name, "- Folder" if item.is_dir() else "- File")

else:
    print("Labeled folder was not found.")

# ===================================================================================
#             COUNT IMAGES IN EACH KL GRADE
# ====================================================================================

# Image formats we want to count
image_extensions = {".png", ".jpg", ".jpeg"}

# Store the total number of images
total_images = 0

print("\n-----KL Grade Distribution-----")

# Loop through KL grades 0, 1,2 3 and 4
for grade in range(5):

    # create the path to the current KL grade folder
    grade_folder = LABELED_FOLDER / str(grade)

    # Find all  supported images files in this folder
    image_files = [
        file for file in grade_folder.iterdir()
        if file.is_file() and file.suffix.lower() in image_extensions
    ]

    # Count images in the current folder
    image_count = len(image_files)

    # Add this count to the overall total
    total_images += image_count

    # Display the result
    print(f"KL Grade {grade}: {image_count} images")

print("-----------------------------------------")
print("Total labeled images: ", total_images)


#==========================================================================================================
#                 VISUALIZE KL GRADE DISTRIBUTION
#==========================================================================================================

import matplotlib.pyplot as plt

# Kl grades
grades = [0, 1, 2, 3, 4]

# Image counts obtained from our dataset
counts = [932, 428, 615, 306, 71]

# Create the bar chart
plt.figure(figsize=(8, 5))

plt.bar(grades, counts, color="steelblue")

# Add chart title and labels
plt.title("Distribution of Knee Osteoarthritis KL Grades")
plt.xlabel("KL Grade")
plt.ylabel("Number of Images")

# Create a folder for our plots
from pathlib import Path

plot_folder = Path("outputs/plots")
plot_folder.mkdir(parents=True, exist_ok=True)

# Save the chart
plt.tight_layout()
plt.savefig(plot_folder / "kl_grade_distribution.png", dpi=300)

# Show the chart
plt.show()

#=======================================================================
#              CHECK IMAGE DIMENSIONS AND FORMATS
#========================================================================

from PIL import Image
from collections import Counter

# Store the different image sizes and formats
image_sizes = Counter()
image_formats = Counter()
image_modes = Counter()

# Loop through all five KL grade folders
for grade in range(5):
    grade_folder = LABELED_FOLDER / str(grade)

    # FInd supported image files
    for file in grade_folder.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:

            # Open the images
            with Image.open(file) as img:

                # Count image dimensions: (width, height)
                image_sizes[img.size] += 1

                # Count image formats: PNG, JPEG, JPG
                image_formats[img.format] += 1

                # Count image modes: L, RGB, RGBA
                image_modes[img.mode] += 1

# Display the results
print("\n--------- Image Dimensions-----------------")
for size, count in image_sizes.items():
    print(f"{size}: {count} images")

print("\n------------Image Formats---------------------")
for image_format, count in image_formats.items():
    print(f"{image_format}: {count} images")

print("\n------------ Image Modes-----------------------")
for mode, count in image_modes.items():
    print(f"{mode}: {count} images")


#============================================================================
#                  CHECKING FOR DUPLICATE IMAGES
#=============================================================================
import hashlib
from collections import defaultdict

# Store image hashes and their file paths
image_hashes = defaultdict(list)

# Loop through KL grades 0 to 4
for grade in range(5):
    grade_folder = LABELED_FOLDER / str(grade)

    # Check each image in the current folder
    for file in grade_folder.iterdir():

        if file.is_file() and file.suffix.lower() in image_extensions:

            # Read  the image file as binary data
            with open(file, "rb") as image_file:
                image_data = image_file.read()

            # Calculate a unique fingerprint of the file contents
            image_hash = hashlib.sha256(image_data).hexdigest()

            # Store the file path under its hash
            image_hashes[image_hash].append(file)

# Find hashes associated with more than one image
duplicates = {
    image_hash: paths
    for image_hash, paths in image_hashes.items()
    if len(paths) > 1
}

# Display the results
print("\n--------- Duplicate Image Check-------------")
print("Total images checked: ", sum(len(paths) for paths in image_hashes.values()))
print("Unique file hashes: ", len(image_hashes))
print("Duplicate groups: ", len(duplicates))

# Print any duplicates found
for image_hash, paths in duplicates.items():
    print("\nDuplicate group:")

    for path in paths:
        print(path)



# ==============================================================================
#               IMAGE PREPROCESSING AND NORMALIZATION
# ===============================================================================

import numpy as np
from PIL import Image

# Create normalization function
def normalize_image(image_array):
    """
    Convert an 8-bit grayscale image into float32
    and normalize its pixel values to the range 0-1
    """
    normalized_image = image_array.astype(np.float32) / 255.0
    return normalized_image

#------------------- Select a sample image------------------
# select KL grade 0 folder
grade_zero_folder = LABELED_FOLDER / "0"

# Find the first PNG image
sample_image_path = sorted(grade_zero_folder.glob("*.png"))[0]

#--------------- Read the Original Image-----------------------------
# Open the image and convert it into a Numpy array
with Image.open(sample_image_path) as img:
    image_array = np.array(img)

# ------------------- Display original Pixel information-------------
print("\n----- Original Image Pixel Information -----")
print("Image:", sample_image_path.name)
print("Shape:", image_array.shape)
print("Data type:", image_array.dtype)
print("Minimum pixel value:", image_array.min())
print("Maximum pixel value:", image_array.max())
print("Mean pixel value:", image_array.mean())

#----------- Normalize the image------------------
# call our resuable normalization function
normalized_image = normalize_image(image_array)

# ------------------ Display normalized pixel information----------
print("\n----- Normalized Image Pixel Information -----")
print("Data type:", normalized_image.dtype)
print("Minimum pixel value:", normalized_image.min())
print("Maximum pixel value:", normalized_image.max())
print("Mean pixel value:", normalized_image.mean())



# =========================================================================
#         VISUAL VERIFICATION OF NORMALIZATION
# =======================================================================

# Create a figure containing two images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Display the original grayscale image
axes[0].imshow(image_array, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original X-ray (0–255)")
axes[0].axis("off")

# Display the normalized grayscale image
axes[1].imshow(normalized_image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Normalized X-ray (0–1)")
axes[1].axis("off")

# Adjust spacing between images
plt.tight_layout()

# Save the comparison in our existing plots folder
plt.savefig(
    plot_folder / "normalization_comparison.png",
    dpi=300
)

# Display the comparison
plt.show()

# Close the figure
plt.close(fig)
