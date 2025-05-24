import os
import json
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
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

# ========== 初始化 DINOv2 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = Dinov2Model.from_pretrained(model_name).to(device)
model.eval()

# ========== 工具函数 ==========
def extract_dino_features(img):
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[0, 1:, :]  # (196, 768)
    return features.cpu().numpy()

def map_coords_to_patch(coords, img_size=(224, 224), patch_grid=14):
    h, w = img_size
    coords = np.array(coords).astype(np.float32)
    grid_x = np.floor(coords[:, 0] / w * patch_grid).astype(int)
    grid_y = np.floor(coords[:, 1] / h * patch_grid).astype(int)
    patch_ids = grid_y * patch_grid + grid_x
    return np.clip(patch_ids, 0, patch_grid * patch_grid - 1)

def cos_sim(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# ========== 主处理流程 ==========
def main():
    args = parse_args()

    os.makedirs(args.click_save_folder, exist_ok=True)

    # 加载平均特征
    with open(args.feature_json_path, "r") as f:
        avg_features = {k: np.array(v) for k, v in json.load(f).items()}
    target_class = args.target_class
    target_class_avg = avg_features[target_class]

    for fname in os.listdir(args.match_txt_folder):
        if not fname.endswith(".txt"):
            continue

        stem = os.path.splitext(fname)[0]
        match_txt_path = os.path.join(args.match_txt_folder, fname)
        img_path = os.path.join(args.image_folder, f"{stem}.jpg")
        if not os.path.exists(img_path):
            continue

        # 1. 读取坐标
        coords = []
        with open(match_txt_path, "r") as f:
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

        # 2. 特征提取
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_np = np.array(img)
        features = extract_dino_features(img)
        patch_ids = map_coords_to_patch(coords)
        point_feats = features[patch_ids]

        # 3. KMeans 聚类
        k = min(5, len(point_feats))
        kmeans = KMeans(n_clusters=k, n_init='auto').fit(point_feats)
        cluster_labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        # 4. 每个聚类中心与 avg_feature 相似度比较
        cluster_to_label = {}
        for i, center in enumerate(cluster_centers):
            sims = {cls: cos_sim(center, v) for cls, v in avg_features.items()}
            best_cls = max(sims, key=sims.get)
            cluster_to_label[i] = target_class if best_cls == target_class else "other"

        # 5. 给每个点分配 label
        final_labeled_points = []
        for i, coord in enumerate(coords):
            cluster_id = cluster_labels[i]
            label = 1 if cluster_to_label[cluster_id] == target_class else 0
            final_labeled_points.append((coord[0], coord[1], label))

        # 6. 保存结果
        with open(os.path.join(args.click_save_folder, f"{stem}.txt"), "w") as f:
            for x, y, label in final_labeled_points:
                f.write(f"{x:.2f},{y:.2f},{label}\n")

        # 7. 可视化保存
        vis_img = img_np.copy()
        for x, y, label in final_labeled_points:
            x, y = int(x), int(y)
            color = [0, 0, 255] if label == 1 else [255, 0, 0]
            vis_img[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = color
        plt.imsave(os.path.join(args.click_save_folder, f"{stem}_vis.png"), vis_img)

    print(f"✅ 聚类匹配完成，结果保存在 {args.click_save_folder}/")

if __name__ == "__main__":
    main()
