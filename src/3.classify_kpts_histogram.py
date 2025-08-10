import os
import json
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Histogram matching classification")
    parser.add_argument('--image_folder', required=True, help='Folder with query images')
    parser.add_argument('--match_txt_folder', required=True, help='Folder with matched keypoints txt files')
    parser.add_argument('--support_img_path', required=True, help='Folder with support images')
    parser.add_argument('--support_mask_path', required=True, help='Folder with support masks')
    parser.add_argument('--click_save_folder', required=True, help='Output folder for click classification')
    parser.add_argument('--target_class', required=True, help='Target class name')
    parser.add_argument('--label_map_json', required=True, help='Path to suv.json file')
    return parser.parse_args()

def load_color_map(json_path):
    with open(json_path, 'r') as f:
        raw = json.load(f)
    return {eval(k): v for k, v in raw.items()}

def extract_histogram(img, point, win_size=11):
    x, y = int(point[0]), int(point[1])
    half = win_size // 2
    patch = img[max(0, y-half):y+half+1, max(0, x-half):x+half+1]
    hist = cv2.calcHist([patch], [0,1,2], None, [8,8,8], [0,256]*3)
    cv2.normalize(hist, hist)
    return hist.flatten()

def classify_point(hist_query, support_imgs, support_masks, color_map, target_class):
    best_sim = -1
    match_class = None
    for img_path, mask_path in zip(support_imgs, support_masks):
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        if img is None or mask is None:
            continue
        h, w = mask.shape[:2]
        for y in range(0, h, 10):
            for x in range(0, w, 10):
                color = tuple(mask[y, x])
                if color not in color_map:
                    continue
                label = color_map[color]
                hist_support = extract_histogram(img, (x, y))
                sim = cv2.compareHist(hist_query, hist_support, cv2.HISTCMP_CORREL)
                if sim > best_sim:
                    best_sim = sim
                    match_class = label
    return int(match_class == target_class)

def main():
    args = parse_args()
    os.makedirs(args.click_save_folder, exist_ok=True)
    color_map = load_color_map(args.label_map_json)

    support_imgs = sorted(glob(os.path.join(args.support_img_path, "*.jpg")))
    support_masks = sorted(glob(os.path.join(args.support_mask_path, "*.png")))

    for txt_path in tqdm(glob(os.path.join(args.match_txt_folder, "*.txt"))):
        img_id = os.path.basename(txt_path).replace(".txt", "")
        img_path = os.path.join(args.image_folder, f"{img_id}.jpg")
        image = cv2.imread(img_path)
        if image is None:
            continue

        with open(txt_path, 'r') as f:
            points = [list(map(float, line.strip().split(','))) for line in f if ',' in line]

        results = []
        for pt in points:
            hist_query = extract_histogram(image, pt)
            is_positive = classify_point(hist_query, support_imgs, support_masks, color_map, args.target_class)
            results.append((*pt, is_positive))

        save_path = os.path.join(args.click_save_folder, f"{img_id}.txt")
        with open(save_path, 'w') as f:
            for x, y, label in results:
                f.write(f"{x:.2f},{y:.2f},{label}\n")

if __name__ == "__main__":
    main()
