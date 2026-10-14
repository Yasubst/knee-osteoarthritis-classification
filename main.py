#--------------------- Read the DICOM file--------------------------------
from hmac import digest_size
from pathlib import Path
import pydicom
from matplotlib.pyplot import viridis

# Location of the folder containing our DICOM images
dicom_folder = Path("data/ROI_Samples/dicoms")

# Find the files inside the folder
dicom_files = sorted(
    file for file in dicom_folder.iterdir() if file.is_file()
)

# Read the first DICOM image
dicom = pydicom.dcmread(dicom_files[0])

# Display basic information
print("DICOM file:", dicom_files[0].name)
print("Image dimensions:", dicom.pixel_array.shape)
print("Pixel spacing:", dicom.get("PixelSpacing", "Not available"))

#--------------------------- Display our first X-ray-------------------------------
import matplotlib.pyplot as plt

#Get the image pixels from the DICOM file
image = dicom.pixel_array

# Display the x-ray in grayscale
plt.imshow(image, cmap="gray")

# Add a title
plt.title("Original Knee X-ray")

# show the image
plt.show()

#----------------------- Understand the BoneFinder landmark file-------------
# Path to the landmark file for DICOM image 1
landmark_path = Path("data/ROI_Samples/landmarks/1.pts")

# open the file and read its contents
with open(landmark_path, "r") as file:
    landmark_content = file.read()

# Print the contents
print("\nBoneFinder landmark file:")
print(landmark_content)

#---------------------------Draw a landmark in X-ray-----------------
# Extract X and Y coordinates from the landmark text
x_coordinates = []
y_coordinates = []

for line in landmark_content.splitlines():
    parts = line.strip().split()

    if len(parts) == 2:
        try:
            x = float(parts[0])
            y = float(parts[1])

            x_coordinates.append(x)
            y_coordinates.append(y)

        except ValueError:
            continue

# Check the number of points
print("Number of landmarks:", len(x_coordinates))

# Display the X-ray with landmarks
plt.figure(figsize=(12, 8))

plt.imshow(image, cmap="gray")

plt.scatter(
    x_coordinates,
    y_coordinates,
    color="red",
    s=8 # controls the marker area so the dots appear small.
)

plt.title("knee X-ray with BoneFinder Landmarks")
plt.show()

#-------------------------Find the bounding box for each knee--------------------
# Separate the landmark coordinates for each knee
right_x = x_coordinates[:74]
right_y = y_coordinates[:74]

left_x = x_coordinates[74:]
left_y = y_coordinates[74:]

# find the smallest and largest coordinates for each knee
right_x_min = min(right_x)
right_x_max = max(right_x)
right_y_min = min(right_y)
right_y_max = max(right_y)

left_x_min = min(left_x)
left_x_max = max(left_x)
left_y_min = min(left_y)
left_y_max = max(left_y)

# print the bounding box coordinates
print("Image-right knee bounding box:")
print(right_x_min, right_y_min, right_x_max, right_y_max)

print("\nImage-left knee bounding box:")
print(left_x_min, left_y_min, left_x_max, left_y_max)

# -------------------Adding 1cm (10mm) margin in each bounding box in all direction-----
import math

# PixelSpacing stores [row spacing, column spacing]
row_spacing = float(dicom.PixelSpacing[0])
column_spacing = float(dicom.PixelSpacing[1])

# Convert the required 1cm margin to pixels
margin_y = math.ceil(10/ row_spacing)
margin_x = math.ceil(10 / column_spacing)

print("Horizontal margin:", margin_x, "pixels")
print("Vertical margin:", margin_y, "pixels")

# Image dimensions
height, width = image.shape

# Calculate expanded bounding boxes
right_box = (
    max(0, math.floor(right_x_min) - margin_x),
    max(0, math.floor(right_y_min) - margin_y),
    min(width, math.ceil(right_x_max) + margin_x),
    min(height, math.ceil(right_y_max) + margin_y)
)

left_box = (
    max(0, math.floor(left_x_min) - margin_x),
    max(0, math.floor(left_y_min) - margin_y),
    min(width, math.ceil(left_x_max) + margin_x),
    min(height, math.ceil(left_y_max) + margin_y)
)

print("Image-right knee ROI:", right_box)
print("Image-left knee ROI:", left_box)