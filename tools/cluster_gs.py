import os
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Function to read coordinates from a text file
def read_roi_coordinates(txt_file):
    with open(txt_file, 'r') as file:
        coords = []
        lines = file.readlines()  # 读取所有行
        for line in lines:
            # 每行的坐标转换为 float，并将其添加到 coords 列表
            coord = list(map(float, line.strip().split()))
            coords.append(coord)
    return coords  # 返回包含所有行的坐标列表

# Function to process each image
def process_image(image_path, roi_txt_path, output_image_folder):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Load the ROI coordinates from a corresponding text file
    roi_coords = read_roi_coordinates(roi_txt_path)

    # Initialize mask (same size as the image) with all zeros
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Process each ROI
    for coord in roi_coords:
        x_min, y_min, x_max, y_max = map(int, coord)

        # Extract the ROI using the coordinates
        roi = image[y_min:y_max, x_min:x_max]

        # Reshape the ROI to a 2D array of pixels
        pixels = roi.reshape(-1, 3)

        # Apply Gaussian Mixture Models clustering
        n_components = 2  # Number of mixture components (clusters)
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(pixels)

        # Get the clustered labels
        labels = gmm.predict(pixels)

        # Create an output image with clustered colors
        clustered_roi = np.zeros_like(roi)

        # Assign colors to the clusters in the output image
        unique_labels = set(labels)
        for label in unique_labels:
            mask_label = (labels == label)
            color = np.random.randint(0, 255, size=3)  # Random color for the cluster
            clustered_roi[mask_label.reshape(roi.shape[0], roi.shape[1])] = color

        # Replace the ROI in the original image with the clustered ROI
        output_image = image.copy()
        output_image[y_min:y_max, x_min:x_max] = clustered_roi

        # Update the binary mask within the ROI based on cluster
        roi_mask = np.zeros_like(roi[:, :, 0])  # Initialize ROI mask (2D)
        for label in unique_labels:
            mask_label = (labels == label)
            value = 255 if label == 0 else 0  # One cluster to white (255), the other to black (0)
            roi_mask[mask_label.reshape(roi.shape[0], roi.shape[1])] = value

        # Place the binary ROI mask into the larger image mask
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
output_image_folder = "./Datasets/CLU/MS011_CLUSTER"

# Process all images in the folder
process_all_images(image_folder, roi_folder, output_image_folder)
