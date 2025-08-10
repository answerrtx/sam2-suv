import os
import json
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Classify keypoints using soft attention and DINOv2")
    parser.add_argument("--image_folder", required=True, help="Folder with original images")
    parser.add_argument("--match_txt_folder", required=True, help="Folder with matched keypoints txt files")
    parser.add_argument("--feature_json_folder", required=True, help="Folder containing multiple support jsons (one per image)")
    parser.add_argument("--click_save_folder", required=True, help="Where to save labeled clicks and visualizations")
    parser.add_argument("--target_class", required=True, help="Target class name for classification")
    return parser.parse_args()

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

def load_support_features(folder):
    support_features = {}
    for fname in os.listdir(folder):
        if not fname.endswith(".json") or "all" in fname:
            continue
        with open(os.path.join(folder, fname), "r") as f:
            raw = json.load(f)
            for label, vec in raw.items():
                vec = np.array(vec)
                if label not in support_features:
                    support_features[label] = []
                support_features[label].append(vec)  # 每张图只有一个 vec，不用 extend
    return support_features

def main():
    args = parse_args()
    os.makedirs(args.click_save_folder, exist_ok=True)
    target_label = args.target_class

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Dinov2Model.from_pretrained(model_name).to(device)
    model.eval()

    support_features = load_support_features(args.feature_json_folder)
    if target_label not in support_features:
        raise ValueError(f"[Error] Label '{target_label}' not found in support features keys: {list(support_features.keys())}")

    label_to_index = {label: idx for idx, label in enumerate(support_features)}
    num_classes = len(label_to_index)

    support_vecs = []
    support_labels = []
    for label, feats in support_features.items():
        print(label, np.array(feats).shape)
        for vec in feats:
            vec = vec / np.linalg.norm(vec)
            support_vecs.append(torch.tensor(vec, dtype=torch.float32))
            support_labels.append(label_to_index[label])

    support_feats = torch.stack(support_vecs)
    support_labels = torch.tensor(support_labels)
    support_labels_onehot = F.one_hot(support_labels, num_classes=num_classes).float()

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
        patch_ids = patch_ids[patch_ids < features.shape[0]]  # 避免越界
        if len(patch_ids) == 0:
            print("⚠️ No valid patch IDs, skipping:", fname)
            continue

        selected_feats = features[patch_ids]
        if selected_feats.ndim != 2 or selected_feats.shape[1] != 768:
            print(f"⚠️ Unexpected feature shape {selected_feats.shape}, skipping:", fname)
            continue
        selected_feats = features[patch_ids]

        query_feats = torch.tensor(selected_feats, dtype=torch.float32)
        query_feats = F.normalize(query_feats, dim=1)
        print("query_feats shape:", query_feats.shape)
        print("support_feats shape:", support_feats.shape)
        sim = torch.matmul(query_feats, support_feats.T)
        attn_weights = F.softmax(sim, dim=1)
        pred_probs = torch.matmul(attn_weights, support_labels_onehot)
        target_index = label_to_index[target_label]
        is_positive = pred_probs[:, target_index] > 0.5

        positive_coords = [coords[i] for i, val in enumerate(is_positive) if val]
        negative_coords = [coords[i] for i, val in enumerate(is_positive) if not val]

        save_txt_path = os.path.join(args.click_save_folder, f"{stem}.txt")
        with open(save_txt_path, "w") as f:
            for x, y in positive_coords:
                f.write(f"{x:.2f},{y:.2f},1\n")
            for x, y in negative_coords:
                f.write(f"{x:.2f},{y:.2f},0\n")

        img_np = np.array(img)
        for x, y in positive_coords:
            x, y = int(x), int(y)
            img_np[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = [0, 0, 255]
        for x, y in negative_coords:
            x, y = int(x), int(y)
            img_np[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = [255, 0, 0]

        plt.imsave(os.path.join(args.click_save_folder, f"{stem}_vis.png"), img_np)

    print(f"✅ 所有帧处理完成，结果保存在 {args.click_save_folder}")

if __name__ == "__main__":
    main()