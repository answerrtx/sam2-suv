import os
import json
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model
import matplotlib.pyplot as plt
import torch
import argparse

# ========== 参数解析 ==========
def parse_args():
    parser = argparse.ArgumentParser(description="Classify keypoints using clustering and DINOv2")
    parser.add_argument("--image_folder", required=True, help="Folder with original images")
    parser.add_argument("--match_txt_folder", required=True, help="Folder with matched keypoints txt files")
    parser.add_argument("--feature_json_path", required=True, help="Path to avg feature json")
    parser.add_argument("--click_save_folder", required=True, help="Where to save labeled clicks and visualizations")
    parser.add_argument("--target_class", required=True, help="Target class name for classification")
    return parser.parse_args()

# ========== 特征提取和映射函数 ==========
def extract_dino_features(img, processor, model, device):
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[0, 1:, :]  # shape (196, 768)
    return features.cpu().numpy()

def map_coords_to_patch(coords, img_size=(224, 224), patch_grid=14):
    h, w = img_size
    coords = np.array(coords).astype(np.float32)
    grid_x = np.floor(coords[:, 0] / w * patch_grid).astype(int)
    grid_y = np.floor(coords[:, 1] / h * patch_grid).astype(int)
    patch_ids = grid_y * patch_grid + grid_x
    patch_ids = np.clip(patch_ids, 0, patch_grid * patch_grid - 1)
    return patch_ids

# ========== 主流程 ==========
def main():
    args = parse_args()

    os.makedirs(args.click_save_folder, exist_ok=True)

    target_label = args.target_class 
    # 初始化 DINOv2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Dinov2Model.from_pretrained(model_name).to(device)
    model.eval()

    # 加载平均特征
    with open(args.feature_json_path, "r") as f:
        avg_features = {k: np.array(v) for k, v in json.load(f).items()}

    if target_label not in avg_features:
        raise ValueError(f"[Error] Label '{target_label}' not found in avg_features keys: {list(avg_features.keys())}")

    for fname in os.listdir(args.match_txt_folder):
        if not fname.endswith(".txt"):
            continue

        stem = os.path.splitext(fname)[0]
        match_txt_path = os.path.join(args.match_txt_folder, fname)
        img_path = os.path.join(args.image_folder, f"{stem}.jpg")
        print(f"🔍 Processing: {img_path}")

        if not os.path.exists(img_path):
            print("❌ Image not found, skipping.")
            continue

        # 读取坐标
        with open(match_txt_path, "r") as f:
            coords = []
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    try:
                        x, y = float(parts[0]), float(parts[1])
                        coords.append([x, y])
                    except:
                        continue
        if not coords:
            continue

        img = Image.open(img_path).convert("RGB").resize((224, 224))
        features = extract_dino_features(img, processor, model, device)
        patch_ids = map_coords_to_patch(coords)
        selected_feats = features[patch_ids]

        # 分类
        positive_coords = []
        negative_coords = []

        for i, feat in enumerate(selected_feats):
            feat_norm = feat / np.linalg.norm(feat)
            sim_scores = {k: np.dot(feat_norm, v / np.linalg.norm(v)) for k, v in avg_features.items()}
            best_label = max(sim_scores, key=sim_scores.get)
            if best_label == target_label:
                positive_coords.append(coords[i])
            else:
                negative_coords.append(coords[i])

        # 保存点击文件
        save_txt_path = os.path.join(args.click_save_folder, f"{stem}.txt")
        with open(save_txt_path, "w") as f:
            for x, y in positive_coords:
                f.write(f"{x:.2f},{y:.2f},1\n")
            for x, y in negative_coords:
                f.write(f"{x:.2f},{y:.2f},0\n")

        # 可视化
        img_np = np.array(img)
        for x, y in positive_coords:
            x, y = int(x), int(y)
            img_np[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = [0, 0, 255]  # 蓝
        for x, y in negative_coords:
            x, y = int(x), int(y)
            img_np[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = [255, 0, 0]  # 红

        plt.imsave(os.path.join(args.click_save_folder, f"{stem}_vis.png"), img_np)

    print(f"✅ 所有帧处理完成，结果保存在 {args.click_save_folder}")

if __name__ == "__main__":
    main()
