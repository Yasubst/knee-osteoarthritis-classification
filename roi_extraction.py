
# ============================================================
# KNEE OSTEOARTHRITIS PROJECT
# PART 1: REGION OF INTEREST (ROI) EXTRACTION
#
# Goal:
# Extract two knee regions from each of four DICOM X-rays
# using BoneFinder landmarks and a 1 cm margin.
# ============================================================


# -------------------- 1. IMPORT LIBRARIES --------------------

from pathlib import Path
import math

import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# -------------------- 2. DEFINE FOLDER PATHS --------------------

# Folder containing DICOM files: 1, 2, 3, 4
DICOM_FOLDER = Path("data/ROI_Samples/dicoms")

# Folder containing landmark files: 1.pts, 2.pts, etc.
LANDMARK_FOLDER = Path("data/ROI_Samples/landmarks")

# Folder where cropped knee images will be saved
OUTPUT_FOLDER = Path("outputs/roi_images")

# Create the output folder if it does not exist
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# -------------------- 3. READ A DICOM IMAGE --------------------

def read_dicom(file_path):
    """Read a DICOM file and return its image and pixel spacing."""

    # Open the DICOM file
    dicom = pydicom.dcmread(file_path)

    # Convert DICOM pixel data into a NumPy image array
    image = dicom.pixel_array


    # PixelSpacing is [row spacing, column spacing] in mm
    row_spacing = float(dicom.PixelSpacing[0])
    column_spacing = float(dicom.PixelSpacing[1])

    return image, row_spacing, column_spacing


# -------------------- 4. READ BONEFINDER LANDMARKS --------------------

def read_landmarks(file_path):
    """Read the X and Y coordinates from a BoneFinder .pts file."""

    x_coordinates = []
    y_coordinates = []

    # Open the landmark file
    with open(file_path, "r") as file:

        # Read the file line by line
        for line in file:

            # Remove extra spaces and split into values
            parts = line.strip().split()

            # Coordinate lines should contain exactly 2 values
            if len(parts) == 2:

                try:
                    # Convert the values into decimal numbers
                    x = float(parts[0])
                    y = float(parts[1])

                    # Store the coordinates
                    x_coordinates.append(x)
                    y_coordinates.append(y)

                except ValueError:
                    # Ignore lines that do not contain numbers
                    continue

    # Check that the expected 148 points were found
    if len(x_coordinates) != 148:
        raise ValueError(
            f"{file_path}: Expected 148 landmarks, "
            f"found {len(x_coordinates)}"
        )

    return x_coordinates, y_coordinates


# -------------------- 5. CALCULATE A BOUNDING BOX --------------------

def calculate_bounding_box(x_coordinates, y_coordinates):
    """Find the smallest rectangle containing all given landmarks."""

    # Find the minimum and maximum horizontal positions
    x_min = min(x_coordinates)
    x_max = max(x_coordinates)

    # Find the minimum and maximum vertical positions
    y_min = min(y_coordinates)
    y_max = max(y_coordinates)

    return x_min, y_min, x_max, y_max


# -------------------- 6. ADD A 1 CM MARGIN --------------------

def add_margin(box, row_spacing, column_spacing, image_shape):
    """Expand a bounding box by 10 mm in every direction."""

    # Convert 10 mm into pixels
    # ceil() rounds upward to the next whole pixel
    margin_y = math.ceil(10 / row_spacing)
    margin_x = math.ceil(10 / column_spacing)

    # Get the image dimensions
    height, width = image_shape

    # Unpack the original bounding box
    x_min, y_min, x_max, y_max = box

    # Expand the rectangle while staying inside the image
    expanded_box = (
        max(0, math.floor(x_min) - margin_x),
        max(0, math.floor(y_min) - margin_y),
        min(width, math.ceil(x_max) + margin_x),
        min(height, math.ceil(y_max) + margin_y)
    )

    return expanded_box


# -------------------- 7. CROP A KNEE REGION --------------------

def crop_knee(image, box):
    """Crop an image using its bounding box coordinates."""

    # Unpack the bounding box
    x_min, y_min, x_max, y_max = box

    # NumPy slicing uses [rows, columns], meaning [Y, X]
    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image


# -------------------- 8. DISPLAY LANDMARKS AND ROI BOXES --------------------

def display_roi(image, x_coordinates, y_coordinates,
                image_left_box, image_right_box, image_number):
    """Display an X-ray with landmarks and both ROI rectangles."""

    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Display the original X-ray
    ax.imshow(image, cmap="gray")

    # Show the BoneFinder landmark points in red
    ax.scatter(
        x_coordinates,
        y_coordinates,
        color="red",
        s=8
    )

    # Draw both bounding boxes using different colors
    for box, color in [
        (image_left_box, "cyan"),
        (image_right_box, "lime")
    ]:

        # Unpack bounding box coordinates
        x_min, y_min, x_max, y_max = box

        # Create the rectangle
        rectangle = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor=color,
            facecolor="none"
        )

        # Add the rectangle to the image
        ax.add_patch(rectangle)

    # Add a descriptive title
    ax.set_title(
        f"Image {image_number}: BoneFinder Landmarks and Knee ROIs"
    )

    # Hide axis numbers
    ax.axis("off")

    # Display the result
    plt.show()

    # Close the figure to release memory
    plt.close(fig)


# -------------------- 9. PROCESS ONE DICOM IMAGE --------------------

def process_image(image_number):
    """Process one X-ray and save its two knee ROI images."""

    print(f"\n---------- Processing image {image_number} ----------")

    # Step A: Build matching file paths
    dicom_path = DICOM_FOLDER / str(image_number)
    landmark_path = LANDMARK_FOLDER / f"{image_number}.pts"

    # Check that both files exist
    if not dicom_path.is_file():
        raise FileNotFoundError(f"Missing DICOM: {dicom_path}")

    if not landmark_path.is_file():
        raise FileNotFoundError(f"Missing landmarks: {landmark_path}")

    # Step B: Read the X-ray
    image, row_spacing, column_spacing = read_dicom(dicom_path)

    # Step C: Read the landmarks
    x_coordinates, y_coordinates = read_landmarks(landmark_path)

    # Step D: Separate the two groups of 74 landmarks
    first_x = x_coordinates[:74]
    first_y = y_coordinates[:74]

    second_x = x_coordinates[74:]
    second_y = y_coordinates[74:]

    # Step E: Calculate a bounding box for each group
    first_box = calculate_bounding_box(first_x, first_y)
    second_box = calculate_bounding_box(second_x, second_y)

    # Step F: Add the required 1 cm margin
    first_box = add_margin(
        first_box,
        row_spacing,
        column_spacing,
        image.shape
    )

    second_box = add_margin(
        second_box,
        row_spacing,
        column_spacing,
        image.shape
    )

    # Step G: Identify image-left and image-right by X position
    # This does not assume the patient's anatomical side.
    first_center_x = (first_box[0] + first_box[2]) / 2
    second_center_x = (second_box[0] + second_box[2]) / 2

    if first_center_x < second_center_x:
        image_left_box = first_box
        image_right_box = second_box
    else:
        image_left_box = second_box
        image_right_box = first_box

    # Step H: Print the calculated ROI information
    print("Image shape:", image.shape)
    print("Pixel spacing:", row_spacing, column_spacing)
    print("Landmarks:", len(x_coordinates))
    print("Image-left ROI:", image_left_box)
    print("Image-right ROI:", image_right_box)

    # Step I: Display the X-ray with ROI rectangles
    # Close the plot window to continue to the next image.
    display_roi(
        image,
        x_coordinates,
        y_coordinates,
        image_left_box,
        image_right_box,
        image_number
    )

    # Step J: Crop both knees
    image_left_knee = crop_knee(image, image_left_box)
    image_right_knee = crop_knee(image, image_right_box)

    # Step K: Save both cropped images
    # Filenames describe image position, not patient anatomy.
    left_output = OUTPUT_FOLDER / f"{image_number}_image_left.png"
    right_output = OUTPUT_FOLDER / f"{image_number}_image_right.png"

    plt.imsave(left_output, image_left_knee, cmap="gray")
    plt.imsave(right_output, image_right_knee, cmap="gray")

    print("Saved:", left_output)
    print("Saved:", right_output)


# -------------------- 10. PROCESS ALL FOUR IMAGES --------------------

def main():
    """Run the complete ROI extraction workflow."""

    # Process DICOM images 1, 2, 3, and 4
    for image_number in range(1, 5):
        process_image(image_number)

    print("\nROI extraction completed!")
    print("Expected output: 8 cropped knee PNG images.")


# -------------------- 11. START THE PROGRAM --------------------

if __name__ == "__main__":
    main()
