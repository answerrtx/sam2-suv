import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to read coordinates from a text file
def read_roi_coordinates(txt_file):
    with open(txt_file, 'r') as file:
        coords = list(map(float, file.readline().strip().split()))
    return coords  # Return as (x_min, y_min, x_max, y_max)

# Function to process each image
def process_image(image_path, roi_txt_path, output_image_folder):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Load the ROI coordinates from a corresponding text file
    roi_coords = read_roi_coordinates(roi_txt_path)

    # Extract the ROI using the coordinates
    x_min, y_min, x_max, y_max = map(int, roi_coords)
    roi = image[y_min:y_max, x_min:x_max]

    # Reshape the ROI to a 2D array of pixels
    pixels = roi.reshape(-1, 3)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(pixels)

    # Get the clustered labels
    labels = kmeans.labels_

    # Create an output image with clustered colors
    clustered_roi = np.zeros_like(roi)

    # Assign yellow to cluster 0 and blue to cluster 1
    yellow = [41, 128, 185]  # Yellow color in RGB
    blue = [ 252, 243, 207 ]      # Blue color in RGB

    clustered_roi[labels.reshape(roi.shape[0], roi.shape[1]) == 0] = yellow
    clustered_roi[labels.reshape(roi.shape[0], roi.shape[1]) == 1] = blue

    # Replace the ROI in the original image with the clustered ROI
    output_image = image.copy()
    output_image[y_min:y_max, x_min:x_max] = clustered_roi

    # Generate a binary mask: one cluster is white (255), the other is black (0)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Initialize the entire mask as black
    roi_mask = np.zeros_like(roi[:, :, 0])  # Create a local mask for the ROI

    # Assign cluster labels to the mask (255 for one cluster, 0 for the other)
    roi_mask[labels.reshape(roi.shape[0], roi.shape[1]) == 0] = 255  # Cluster 0 as white (255)

    # Copy the ROI mask to the main mask
    mask[y_min:y_max, x_min:x_max] = roi_mask

    # Save the clustered output image
    base_name = os.path.basename(image_path).replace('.jpg', '')
    output_image_path = os.path.join(output_image_folder, f"{base_name}_vis.jpg")
    clustered_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, clustered_image_bgr)

    # Save the binary mask image
    mask_output_path = os.path.join(output_image_folder, f"{base_name}.jpg")
    cv2.imwrite(mask_output_path, mask)

    print(f"Processed and saved: {output_image_path} and {mask_output_path}")

# Main function to iterate through all images in the folder
def process_all_images(image_folder, roi_folder, output_image_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    # Iterate through all image files in the folder
    for image_file in os.listdir(image_folder):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(image_folder, image_file)
            base_name = os.path.splitext(image_file)[0]
            roi_txt_path = os.path.join(roi_folder, f"{base_name}.txt")

            # Check if corresponding ROI file exists
            if os.path.exists(roi_txt_path):
                process_image(image_path, roi_txt_path, output_image_folder)
            else:
                print(f"ROI file not found for: {image_file}")

# Paths
image_folder = "./Datasets/CLU/MS011"
roi_folder = "./Datasets/CLU/MS011_ROIS_AF"
output_image_folder = "./Datasets/CLU/MS011_CLUSTER2"

# Process all images in the folder
process_all_images(image_folder, roi_folder, output_image_folder)
