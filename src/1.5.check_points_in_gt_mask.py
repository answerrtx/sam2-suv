# src/point_mask_check.py

import os
import json
import argparse
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm

def parse_color_map(suv_json_path):
    with open(suv_json_path, "r") as f:
        data = json.load(f)
    return {eval(k): v for k, v in data.items()}  # 转为 {tuple(R, G, B): class}

def get_colors_for_class(target_class, color_map):
    return [color for color, label in color_map.items() if label == target_class]

def read_click_txt(click_path):
    #print(click_path)
    points = []
    with open(click_path, "r") as f:
        for line in f:
            #print(line)
            parts = line.strip().split(',')
            if len(parts) >= 2:
                #print(parts[2])
                x, y, label = int(float(parts[0])), int(float(parts[1])), int((parts[2]))
                points.append((x, y, label))
    return points

def check_points_against_class_mask(base_dataset_path, folder_name, click_folder, suv_json_path, class_name, save_path=None):
    color_map = parse_color_map(suv_json_path)
    class_colors = get_colors_for_class(class_name, color_map)
    mask_dir = os.path.join(base_dataset_path, folder_name, "mask")
    print(click_folder)
    click_files = sorted(glob(os.path.join(click_folder, "*.txt")))

    result = {}

    for click_file in tqdm(click_files, desc=f"Checking {class_name} points"):
        fname = os.path.basename(click_file).replace(".txt", "")
        mask_path = os.path.join(mask_dir, f"{fname}_mask.png")
        if not os.path.exists(mask_path):
            print(f"Warning: Missing mask for {fname}")
            continue

        mask = np.array(Image.open(mask_path).convert("RGB"))
        click_points = read_click_txt(click_file)
        #print(click_file,len(click_points))

        hit_count = 0
        total_count = 0
        correct_count = 0
        total_pos = 0
        TP = FP = TN = FN = 0

        for x, y, label in click_points:
            total_count +=1
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                pixel = tuple(mask[y, x])
                total_count += 1
                if label == 1:
                    if pixel in class_colors:
                        hit_count += 1
                        TP+=1
                    else:
                        FP+=1
                else:
                    if pixel in class_colors:
                        hit_count+= 1
                        FN+=1
                    else:
                        TN+=1

        total = TP + FP + TN + FN
        accuracy = (TP + TN) / total if total > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        print(f"Total: {total_count}, TP: {TP}, hit: {hit_count}, FP: {FP}, TN: {TN}, FN: {FN}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        result[fname] = {
            "total": total_count,
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
        }
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved result to {save_path}")
    else:
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check class-specific points against GT mask")
    parser.add_argument("--base_dataset_path", required=True)
    parser.add_argument("--folder_name", required=True)
    parser.add_argument("--click_folder", required=True)
    parser.add_argument("--suv_json_path", required=True)
    parser.add_argument("--class_name", required=True)
    parser.add_argument("--save_path", help="Optional JSON output")

    args = parser.parse_args()
    print(args.folder_name, args.click_folder, args.suv_json_path, args.class_name)
    check_points_against_class_mask(
        args.base_dataset_path,
        args.folder_name,
        args.click_folder,
        args.suv_json_path,
        args.class_name,
        args.save_path
    )
