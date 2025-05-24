"""
python sample_points_from_masks.py \
  --img_folder imgs/ \
  --mask_folder masks/ \
  --config config.json \
  --out_folder sampled_txts/ \
  --vis_folder sampled_vis/

python 1.sample_pt_from_mask.py --img_folder Datasets/MSKUSO/support/hand_wrist_imgs --mask_folder Datasets/MSKUSO/support/hand_wrist_mask --config ./suv.json --out_folder Datasets/MSKUSO/support/hand_wrist_rnd_pt --vis_folder Datasets/MSKUSO/support/hand_wrist_rnd_vis
"""


import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/acceon/anaconda3/envs/sam2suv/lib/python3.10/site-packages/cv2/qt/plugins/platforms'

import json
import numpy as np
from PIL import Image
from collections import defaultdict
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
def load_config(config_path):
    with open(config_path, 'r') as f:
        color_map = json.load(f)
    # Convert string keys like "(255, 0, 0)" to tuple keys
    color_map = {eval(k): v for k, v in color_map.items()}
    return color_map


def sample_points_from_mask(mask, color_map, min_points=10, max_points=15, bg_points=10):
    mask_np = np.array(mask)
    h, w, _ = mask_np.shape if mask_np.ndim == 3 else (*mask_np.shape, 1)
    label_points = defaultdict(list)

    used_mask = np.zeros((h, w), dtype=bool)

    for color, label in color_map.items():
        match = np.all(mask_np == color, axis=-1).astype(np.uint8)

        # 防止过小区域腐蚀后消失
        if match.sum() < 20:
            core_area = match
        else:
            kernel = np.ones((3, 3), np.uint8)
            core_area = cv2.erode(match, kernel, iterations=1)

        indices = np.argwhere(core_area == 1)
        if len(indices) == 0:
            continue

        n_sample = random.randint(min_points, max_points)
        sampled = indices[np.random.choice(len(indices), min(n_sample, len(indices)), replace=False)]
        label_points[label].extend(sampled.tolist())
        used_mask |= (match == 1)

    # 背景采样：避免腐蚀后的前景区域
    bg_mask = (~used_mask).astype(np.uint8)
    if bg_mask.sum() > 0:
        bg_indices = np.argwhere(bg_mask)
        sampled_bg = bg_indices[np.random.choice(len(bg_indices), min(bg_points, len(bg_indices)), replace=False)]
        label_points["__bg__"].extend(sampled_bg.tolist())

    return label_points


def save_points(label_points, save_path):
    with open(save_path, 'w') as f:
        json.dump(label_points, f, indent=2)


def visualize_sampled_points(mask, label_points, save_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(mask)
    colors = {
        "__bg__": 'gray',
        # 其他类别可以扩展颜色表或随机分配颜色
    }

    for i, (label, points) in enumerate(label_points.items()):
        points = np.array(points)
        if label not in colors:
            colors[label] = plt.cm.tab10(i % 10)
        if len(points) > 0:
            plt.scatter(points[:, 1], points[:, 0], s=8, c=[colors[label]], label=label, alpha=0.8)

    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path.replace("_mask",""), dpi=150)
    plt.close()


def main(img_folder, mask_folder, config_path, out_folder, vis_folder=None):
    os.makedirs(out_folder, exist_ok=True)
    print(mask_folder)
    if vis_folder:
        os.makedirs(vis_folder, exist_ok=True)

    color_map = load_config(config_path)

    for fname in tqdm(os.listdir(mask_folder)):
        #print(fname)
        if not fname.endswith(('.png', '.jpg', '.jpeg')):
            continue
        mask_path = os.path.join(mask_folder, fname)
        mask = Image.open(mask_path).convert("RGB")
        label_points = sample_points_from_mask(mask, color_map)
        save_path = os.path.join(out_folder, os.path.splitext(fname)[0] + ".json")
        save_points(label_points, save_path.replace("_mask",""))

        if vis_folder:
            vis_path = os.path.join(vis_folder, os.path.splitext(fname)[0] + "_vis.png")
            visualize_sampled_points(mask, label_points, vis_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', required=True, help='Path to image folder (not used directly but kept for completeness)')
    parser.add_argument('--mask_folder', required=True, help='Path to mask folder')
    parser.add_argument('--config', required=True, help='Path to config.json with color-label mapping')
    parser.add_argument('--out_folder', required=True, help='Folder to save sampled points txt files')
    parser.add_argument('--vis_folder', help='Optional folder to save visualizations of sampled points', default=None)
    args = parser.parse_args()

    main(args.img_folder, args.mask_folder, args.config, args.out_folder,args.vis_folder)
