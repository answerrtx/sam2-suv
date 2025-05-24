import os
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from medpy.metric.binary import hd95
from skimage.io import imread
from sklearn.metrics import jaccard_score, f1_score
import pandas as pd


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def calculate_metrics(pred, gt):
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    iou = jaccard_score(gt_flat, pred_flat, zero_division=0)
    dice = f1_score(gt_flat, pred_flat, zero_division=0)
    hd = hd95(pred.astype(np.bool_), gt.astype(np.bool_)) if np.any(pred) and np.any(gt) else np.nan
    return iou, dice, hd


def evaluate_per_folder(results_base, dataset_base, region_map, color_map):
    eval_results = {}

    for region, folders in region_map.items():
        region_scores = {}
        for folder in folders:
            pred_mask_dir = os.path.join(results_base, folder, "merged_results")
            gt_mask_dir = os.path.join(dataset_base, folder, "mask")
            pred_mask_paths = sorted(glob(os.path.join(pred_mask_dir, "*.png")))
            folder_scores = {}

            for pred_mask_path in pred_mask_paths:
                filename = os.path.basename(pred_mask_path)
                gt_mask_path = os.path.join(gt_mask_dir, filename)
                if not os.path.exists(gt_mask_path):
                    continue

                pred_mask = imread(pred_mask_path)
                gt_mask = imread(gt_mask_path)

                class_scores = {}
                for color_str, class_name in color_map.items():
                    color = tuple(map(int, color_str.strip("()").split(',')))
                    pred_bin = np.all(pred_mask[:, :, :3] == color, axis=-1).astype(np.uint8)
                    gt_bin = np.all(gt_mask[:, :, :3] == color, axis=-1).astype(np.uint8)
                    iou, dice, hd = calculate_metrics(pred_bin, gt_bin)
                    if class_name not in class_scores:
                        class_scores[class_name] = {'iou': [], 'dice': [], 'hd95': []}
                    class_scores[class_name]['iou'].append(iou)
                    class_scores[class_name]['dice'].append(dice)
                    class_scores[class_name]['hd95'].append(hd)

                for cls, scores in class_scores.items():
                    if cls not in folder_scores:
                        folder_scores[cls] = {'iou': [], 'dice': [], 'hd95': []}
                    for k in scores:
                        folder_scores[cls][k].append(np.nanmean(scores[k]))

            # average across video
            for cls, scores in folder_scores.items():
                if cls not in region_scores:
                    region_scores[cls] = {'iou': [], 'dice': [], 'hd95': []}
                for k in scores:
                    region_scores[cls][k].append(np.nanmean(scores[k]))

        # average across region
        for cls, scores in region_scores.items():
            for k in scores:
                region_scores[cls][k] = np.nanmean(scores[k])

        eval_results[region] = region_scores
    return eval_results


# Load JSONs
suv_json = load_json("suv.json")
region_json = load_json("region.json")

# Generate color map
color_map = {k: v for k, v in suv_json['color_map'].items()}
region_map = region_json  # assumes format {region: [folder1, folder2, ...]}

# Paths
results_base = "results"
dataset_base = "Dataset"

# Run evaluation
evaluation = evaluate_per_folder(results_base, dataset_base, region_map, color_map)

# Convert to DataFrame for display
records = []
for region, cls_data in evaluation.items():
    for cls, metrics in cls_data.items():
        records.append({
            "Region": region,
            "Class": cls,
            "IOU": metrics['iou'],
            "DICE": metrics['dice'],
            "HD95": metrics['hd95'],
        })

df = pd.DataFrame(records)
import ace_tools as tools; tools.display_dataframe_to_user(name="Segmentation Evaluation Summary", dataframe=df)
