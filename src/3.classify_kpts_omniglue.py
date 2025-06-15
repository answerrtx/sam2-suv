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
    parser.add_argument("--ref_img_path", required=True, help="Support image path")
    parser.add_argument("--ref_mask_path", required=True, help="Support mask path")
    parser.add_argument("--color_map_path", required=True, help="Path to color map json")
    parser.add_argument("--click_save_folder", required=True, help="Output folder for click results")
    parser.add_argument("--anchor_file", required=True, help="List of image filenames to process")
    parser.add_argument("--target_class", required=True, help="Target class name for classification")
    parser.add_argument("--subregion", type=str, default=None, required=True)
    parser.add_argument("--kpt_folder", required=False, default=None, help="Folder to load keypoints")
    parser.add_argument("--og_export", type=str, default=None, help="Export OmniGlue results")
    parser.add_argument("--sp_export", type=str, default=None, help="Export SuperPoint results")
    parser.add_argument("--dino_export", type=str, default=None, help="Export DINO results")
    return parser.parse_args()

def load_mask_black_white(mask_path, target_rgb):
    """
    加载彩色 mask 并返回一个黑白二值 mask。
    
    参数:
        mask_path (str): 彩色 mask 文件路径。
        target_rgb (tuple): 目标类别的 RGB 值，例如 (255, 0, 0)。
    
    返回:
        np.ndarray: 黑白二值 mask，目标区域为 255，其余为 0。
    """
    print(mask_path)
    mask_img = np.array(Image.open(mask_path).convert("RGB"))
    binary_mask = np.all(mask_img == np.array(target_rgb), axis=-1).astype(np.uint8) * 255
    return binary_mask

from collections import defaultdict
import cv2
def main():
    args = parse_args()

    os.makedirs(args.click_save_folder, exist_ok=True)
    print(args.color_map_path)
    # === 读取颜色标签映射 ===
    with open(args.color_map_path, "r") as f:
        color_map = json.load(f)
        color_map = {tuple(map(int, k.strip("()").split(","))): v for k, v in color_map.items()}
        mc_color = [k for k, v in color_map.items() if v == "Deltoid"][0]#args.target_class][0]

    # === 加载 support 掩膜并提取 区域 ===
    mask_img = Image.open(args.ref_mask_path).convert("RGB")
    mask_np = np.array(mask_img)
    mc_mask = np.all(mask_np == mc_color, axis=-1)  # shape: (H, W)
    image_folder = os.path.join(args.image_folder, "imgs")
    ref_mask1 = mc_mask.astype(np.uint8) * 255
    #plt.figure(figsize=(6, 6))
    #plt.imshow(ref_mask, cmap='gray')
    #plt.title(title)
    #plt.axis('off')
    #plt.show()

    # === 初始化 OmniGlue ===
    og = omniglue.OmniGlue(
        og_export=args.og_export,
        sp_export=args.sp_export,
        dino_export=args.dino_export,
    )

    # === 读取帧文件名 ===
    with open(args.anchor_file, "r") as f:
        frame_list = [line.strip() for line in f if line.strip().endswith(".jpg")]

    for fname in frame_list:
        stem = os.path.splitext(fname)[0]
        image_path = os.path.join(image_folder, fname)
        if not os.path.exists(image_path):
            print(f"❌ 缺失帧：{fname}")
            continue

        # === 匹配点提取 ===
        image0 = np.array(Image.open(image_path).convert("RGB"))
        image1 = np.array(Image.open(args.ref_img_path).convert("RGB"))

        labels = []
        match_mask = None    
        match_kp0_all = []
        match_kp1_all = []
        conf_all = []
        best_cluster = None
        best_score = -1
        match_kp0_best = None
        match_kp1_best = None
        conf_best = None
        match_kp0_best = [-1]
        
        if args.subregion!=None and args.kpt_folder:
            kpt_path = os.path.join(args.kpt_folder, f"{stem}.txt")
            if os.path.exists(kpt_path):
                cluster_points = defaultdict(list)
                with open(kpt_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        parts = line.strip().split(",")
                        if len(parts) == 3:
                            x, y, cluster = float(parts[0]), float(parts[1]), parts[2].strip()
                            cluster_points[cluster].append([x, y])

                h, w = image0.shape[:2]
                best_mask = None
                for cluster_name, points in cluster_points.items():
                    if len(points) < 3:
                        continue  # 跳过无法构建凸包的cluster
                    cluster_mask = np.zeros((h, w), dtype=np.uint8)
                    print(cluster_name, h, w)
                    pts = np.array(points, dtype=np.int32)
                    hull = cv2.convexHull(pts)
                    cv2.fillConvexPoly(cluster_mask, hull, 255)
                    """plt.figure(figsize=(6, 6))
                    plt.imshow(cluster_mask, cmap='gray')
                    plt.title(f"Cluster: {cluster_name}")
                    plt.axis("off")
                    plt.show()"""
                    match_kp0, match_kp1, confidences = og.FindMatches(
                        image0, image1, match_mask=cluster_mask, ref_mask=ref_mask1
                    )
                    plt.close()
                    keep = confidences > 0.02
                    match_kp0 = match_kp0[keep]
                    match_kp1 = match_kp1[keep]
                    if len(match_kp0) < 2:
                        print(f"⚠️ {fname} Cluster '{cluster_name}' 匹配点过少，跳过。")
                        continue
                    confidences = confidences[keep]
                    conf_mean = confidences.mean()

                    print(f"{fname} Cluster '{cluster_name}': matched {len(match_kp0)} points, "
                        f"confidence mean={confidences.mean():.4f}, median={np.median(confidences):.4f}")
                    
                    if len(match_kp0) > len(match_kp0_best):
                        best_score = conf_mean
                        best_cluster = cluster_name
                        match_kp0_best = match_kp0
                        match_kp1_best = match_kp1
                        conf_best = confidences
                        best_mask = cluster_mask 
                    # region === 可视化匹配 ===
                    mask_rgb = np.stack([cluster_mask]*3, axis=-1)  # shape: (H, W, 3)
                    mask_rgb = np.uint8(mask_rgb)  # ensure type is uint8
                    vis0 = image0.copy()
                    vis1 = image1.copy()
                    vish = max(mask_rgb.shape[0], vis0.shape[0], vis1.shape[0])
                    visw = mask_rgb.shape[1] + vis0.shape[1] + vis1.shape[1]
                    canvas = np.ones((vish, visw, 3), dtype=np.uint8) * 255
                    canvas[:mask_rgb.shape[0], :mask_rgb.shape[1]] = mask_rgb
                    canvas[:vis0.shape[0], mask_rgb.shape[1]:mask_rgb.shape[1] + vis0.shape[1]] = vis0
                    canvas[:vis1.shape[0], mask_rgb.shape[1] + vis0.shape[1]:] = vis1

                    # 偏移量：image0 的起点为 mask 宽度，image1 的起点为 mask+image0 宽度
                    offset0 = mask_rgb.shape[1]
                    offset1 = offset0 + vis0.shape[1]

                    for pt0, pt1, conf in zip(match_kp0, match_kp1, confidences):
                        x0, y0 = int(round(pt0[0] + offset0)), int(round(pt0[1]))
                        x1, y1 = int(round(pt1[0] + offset1)), int(round(pt1[1]))
                        color = tuple(np.random.randint(0, 255, 3).tolist())
                        cv2.line(canvas, (x0, y0), (x1, y1), color, 1)
                        cv2.circle(canvas, (x0, y0), 2, color, -1)
                        cv2.circle(canvas, (x1, y1), 2, color, -1)

                    vis_path = os.path.join(args.click_save_folder, f"{stem}_{cluster_name}_matchvis.png")
                    cv2.imwrite(vis_path, canvas[:, :, ::-1])  # BGR to RGB
                    #match_kp0_all.append(match_kp0)
                    #match_kp1_all.append(match_kp1)
                    #conf_all.append(confidences)
                    
                    # endregion
                
                print(best_cluster,"=====",len(match_kp0_best))
                for pt0, pt1 in zip(match_kp0_best, match_kp1_best):
                    x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
                    label = 1 
                    labels.append((pt0[0], pt0[1], label))
                
                best_outside = (best_mask == 0)
                ys, xs = np.where(best_outside)
                if len(xs) < len(match_kp0):
                    print(f"⚠️ Warning: Not enough outside points to sample for {cluster_name}")
                    sample_indices = np.arange(len(xs))  # 全部使用
                else:
                    sample_indices = np.random.choice(len(xs), size=len(match_kp0_best), replace=False)

                for idx in sample_indices:
                    x_rand, y_rand = xs[idx], ys[idx]
                    labels.append((float(x_rand), float(y_rand), 0))
            else:
                print(f"⚠️ 缺失 cluster 信息文件：{kpt_path}")
                continue
        else:
            # 非 subregion 模式：全图匹配一次
            match_kp0, match_kp1, confidences = og.FindMatches(
                image0, image1, match_mask=None, ref_mask=ref_mask1
            )
            keep = confidences > 0.02
            match_kp0_all = [match_kp0[keep]]
            match_kp1_all = [match_kp1[keep]]
            conf_all = [confidences[keep]]
            print(f"[Global match]: matched {len(match_kp0_all[0])} points, "
                f"confidence mean={conf_all[0].mean():.4f}, median={np.median(conf_all[0]):.4f}")

        
            
            for pt0, pt1 in zip(match_kp0_best, match_kp1_best):
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
