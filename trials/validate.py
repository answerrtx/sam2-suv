import os
import cv2
import numpy as np
from sklearn.metrics import jaccard_score, f1_score


def calculate_metrics(gt, model):
    # Flatten arrays for pixel-wise comparison
    gt_flat = gt.flatten()
    model_flat = model.flatten()
    
    # Calculate IoU
    iou = jaccard_score(gt_flat, model_flat, zero_division=0)
    
    # Calculate Dice Score
    dice = f1_score(gt_flat, model_flat, zero_division=0)
    
    return iou, dice


def process_directory(directory_path):
    """
    Calculate directory-wise mIoU and mDice for all files in a directory.
    """
    print(f"Processing directory: {directory_path}")
    gt_folder = os.path.join(directory_path, 'mask')
    model_folder = os.path.join(directory_path, 'gsam2')

    if not os.path.exists(gt_folder):
        print(f"Ground truth folder not found: {gt_folder}")
        return 0, 0
    if not os.path.exists(model_folder):
        print(f"Model folder not found: {model_folder}")
        return 0, 0

    ious = []
    dices = []

    for gt_file in os.listdir(gt_folder):
        gt_path = os.path.join(gt_folder, gt_file)
        model_mask_name = gt_file.split('_mask')[0] + '.jpg'
        model_path = os.path.join(model_folder, model_mask_name)

        if os.path.exists(model_path):
            # Read and binarize the images
            gt = (cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            model = (cv2.imread(model_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

            # Resize model image to match ground truth (gt) size
            model_resized = cv2.resize(model, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Calculate metrics
            iou, dice = calculate_metrics(gt, model_resized)
            ious.append(iou)
            dices.append(dice)

    # Calculate directory-wise mIoU and mDice
    directory_miou = np.mean(ious) if ious else 0
    directory_mdice = np.mean(dices) if dices else 0
    return directory_miou, directory_mdice


def calculate_classwise_metrics(class_folders):
    """
    Calculate directory-wise and pixel-wise mIoU and mDice for all directories in a class.
    """
    class_results = {}  # Store directory-wise results
    all_gt_pixels = []
    all_model_pixels = []

    for folder in class_folders:
        # Add prefix path
        folder_path = os.path.join('./Datasets/YTU', folder)

        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        # Calculate directory-wise mIoU and mDice
        directory_miou, directory_mdice = process_directory(folder_path)
        class_results[folder] = {'Directory mIoU': directory_miou, 'Directory mDice': directory_mdice}

        # For pixel-wise metrics, accumulate all pixels
        gt_folder = os.path.join(folder_path, 'mask')
        model_folder = os.path.join(folder_path, 'gsam2')

        for gt_file in os.listdir(gt_folder):
            gt_path = os.path.join(gt_folder, gt_file)
            model_mask_name = gt_file.split('_mask')[0] + '.jpg'
            model_path = os.path.join(model_folder, model_mask_name)

            if os.path.exists(model_path):
                # Read and binarize the images
                gt = (cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
                model = (cv2.imread(model_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

                # Resize model image to match ground truth (gt) size
                model_resized = cv2.resize(model, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Flatten and append to global arrays
                all_gt_pixels.extend(gt.flatten())
                all_model_pixels.extend(model_resized.flatten())

    # Convert to numpy arrays for pixel-wise calculation
    all_gt_pixels = np.array(all_gt_pixels, dtype=np.uint8)
    all_model_pixels = np.array(all_model_pixels, dtype=np.uint8)

    # Calculate pixel-wise mIoU and mDice
    pixelwise_miou = jaccard_score(all_gt_pixels, all_model_pixels, zero_division=0)
    pixelwise_mdice = f1_score(all_gt_pixels, all_model_pixels, zero_division=0)

    return class_results, pixelwise_miou, pixelwise_mdice


# Define the paths for each class
hip_folders = ['hp1', 'hp2', 'hp4', 'hp8', 'hp12', 'hp14', 'hp17', 'hp20', 'hp21', 'hp22', 'hp25', 'hp27']
hand_folders = ['hnd1', 'hnd2', 'hnd3', 'hnd4', 'hnd5', 'wst3', 'wst4', 'wst5', 'wst7', 'wst8']
spine_folders = ['sp1', 'sp5', 'sp6', 'sp7', 'sp10', 'sp11', 'sp12', 'sp13']

# Calculate metrics for each class
print("\nCalculating for 'hand' class:")
hand_results, hand_pixelwise_iou, hand_pixelwise_dice = calculate_classwise_metrics(hand_folders)
print("Hand class directory-wise metrics:", hand_results)
print("Hand class pixel-wise mIoU:", hand_pixelwise_iou)
print("Hand class pixel-wise Dice:", hand_pixelwise_dice)

print("\nCalculating for 'hip' class:")
hip_results, hip_pixelwise_iou, hip_pixelwise_dice = calculate_classwise_metrics(hip_folders)
print("Hip class directory-wise metrics:", hip_results)
print("Hip class pixel-wise mIoU:", hip_pixelwise_iou)
print("Hip class pixel-wise Dice:", hip_pixelwise_dice)

print("\nCalculating for 'spine' class:")
spine_results, spine_pixelwise_iou, spine_pixelwise_dice = calculate_classwise_metrics(spine_folders)
print("Spine class directory-wise metrics:", spine_results)
print("Spine class pixel-wise mIoU:", spine_pixelwise_iou)
print("Spine class pixel-wise Dice:", spine_pixelwise_dice)
