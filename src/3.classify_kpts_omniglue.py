import os
import json
import argparse
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model
import matplotlib.pyplot as plt
import torch
#import tensorflow as tf
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omniglue_onnx import omniglue
from omniglue_onnx.omniglue import utils

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#physical_devices = tf.config.list_physical_devices('GPU')
#if physical_devices:
#    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def parse_args():
    parser = argparse.ArgumentParser(description="OmniGlue keypoint matching")
    parser.add_argument("--image_folder", required=True, help="Folder containing query images")
    parser.add_argument("--support_img_path", required=True, help="Support image path")
    parser.add_argument("--support_mask_path", required=True, help="Support mask path")
    parser.add_argument("--color_map_path", required=True, help="Path to color map json")
    parser.add_argument("--click_save_folder", required=True, help="Output folder for click results")
    parser.add_argument("--index_file", required=True, help="List of image filenames to process")
    parser.add_argument("--target_class", required=True, help="Target class name for classification")

    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.click_save_folder, exist_ok=True)

    # === 读取颜色标签映射 ===
    with open(args.color_map_path, "r") as f:
        color_map = json.load(f)
        color_map = {tuple(map(int, k.strip("()").split(","))): v for k, v in color_map.items()}
        mc_color = [k for k, v in color_map.items() if v == args.target_class][0]

    # === 加载 support 掩膜并提取 MC 区域 ===
    mask_img = Image.open(args.support_mask_path).convert("RGB")
    mask_np = np.array(mask_img)
    mc_mask = np.all(mask_np == mc_color, axis=-1)  # shape: (H, W)
    image_folder = os.path.join(args.image_folder, "imgs")
    # === 初始化 OmniGlue ===
    og = omniglue.OmniGlue(
        og_export="./checkpoints/omniglue.onnx",
        sp_export="./checkpoints/sp_v6.onnx",
        dino_export="./checkpoints/dinov2_vitb14_pretrain.pth",
    )

    # === 读取帧文件名 ===
    with open(args.index_file, "r") as f:
        frame_list = [line.strip() for line in f if line.strip().endswith(".jpg")]

    for fname in frame_list:
        stem = os.path.splitext(fname)[0]
        image_path = os.path.join(image_folder, fname)
        if not os.path.exists(image_path):
            print(f"❌ 缺失帧：{fname}")
            continue

        # === 匹配点提取 ===
        image0 = np.array(Image.open(image_path).convert("RGB"))
        image1 = np.array(Image.open(args.support_img_path).convert("RGB"))
        match_kp0, match_kp1, confidences = og.FindMatches(image0, image1)

        keep = confidences > 0.02
        match_kp0 = match_kp0[keep]
        match_kp1 = match_kp1[keep]
        if(len(match_kp0) < 2): continue
        # === 判断 support 点是否在 MC 掩膜区域 ===
        labels = []
        for pt0, pt1 in zip(match_kp0, match_kp1):
            x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
            if 0 <= y1 < mc_mask.shape[0] and 0 <= x1 < mc_mask.shape[1]:
                y_min = max(0, y1 - 5)
                y_max = min(mc_mask.shape[0], y1 + 5)
                x_min = max(0, x1 - 5)
                x_max = min(mc_mask.shape[1], x1 + 5)
                label = 1 if np.any(mc_mask[y_min:y_max, x_min:x_max]) else 0
            else:
                label = 0
            labels.append((pt0[0], pt0[1], label))

        # === 保存结果 ===
        save_txt = os.path.join(args.click_save_folder, f"{stem}.txt")
        with open(save_txt, "w") as f:
            for x, y, label in labels:
                f.write(f"{x:.2f},{y:.2f},{label}\n")

        vis_img = image0.copy()
        for x, y, label in labels:
            x, y = int(x), int(y)
            color = [0, 0, 255] if label == 1 else [255, 0, 0]
            vis_img[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = color
        plt.imsave(os.path.join(args.click_save_folder, f"{stem}_vis.png"), vis_img)

    print("✅ 匹配与标注完成，点击结果已保存。")

if __name__ == "__main__":
    main()
