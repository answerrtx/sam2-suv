import os
import json
import argparse
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
import sys
from sklearn.decomposition import PCA
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omniglue_onnx import omniglue
from omniglue_onnx.omniglue import utils

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# === 初始化 DINOv2 vitb14 ===
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device)
dinov2.eval()
pca_model = PCA(n_components=32)
_pca_fitted = False

def apply_pca(feat, fit_pca=False):
    global pca_model, _pca_fitted
    if fit_pca and not _pca_fitted:
        pca_model.fit(feat)
        _pca_fitted = True
    if _pca_fitted:
        return pca_model.transform(feat)
    else:
        return feat
# === 图像特征提取 ===
def extract_dinov2_feature(image_path, fit_pca=False):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dinov2(**inputs)
        feat = outputs.last_hidden_state[:, 1:, :].squeeze(0).cpu().numpy()  # shape: (196, 768)
    return apply_pca(feat, fit_pca=fit_pca)
    
def cosine_similarity(f1, f2):
    f1 = np.asarray(f1).flatten()
    f2 = np.asarray(f2).flatten()
    #print(len(f1), len(f2),"+++++")
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))

def select_best_support(query_img_path, support_label_dir, support_img_dir, target_class):
    candidate_imgs = []
    for label_file in os.listdir(support_label_dir):
        if label_file.endswith(".txt"):
            with open(os.path.join(support_label_dir, label_file)) as f:
                labels = [line.strip() for line in f]
            if target_class in labels:
                img_file = label_file.replace(".txt", ".jpg")
                candidate_imgs.append(os.path.join(support_img_dir, img_file))
    if not candidate_imgs:
        raise ValueError(f"No support image contains class '{target_class}'")

    query_feat = extract_dinov2_feature(query_img_path)
    best_score, best_img = -float("inf"), None
    for support_path in candidate_imgs:
        support_feat = extract_dinov2_feature(support_path)
        score = cosine_similarity(query_feat, support_feat)
        if score > best_score:
            best_score, best_img = score, support_path
    print(query_img_path,best_img)
    return best_img

def parse_args():
    parser = argparse.ArgumentParser(description="OmniGlue keypoint matching")
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--support_label_dir", required=True)
    parser.add_argument("--support_img_dir", required=True)
    parser.add_argument("--support_mask_dir", required=True)
    parser.add_argument("--color_map_path", required=True)
    parser.add_argument("--click_save_folder", required=True)
    parser.add_argument("--index_file", required=True)
    parser.add_argument("--match_kpt_folder", required=True)
    parser.add_argument("--target_class", required=True)
    return parser.parse_args()

def main():
    _pca_fitted = False

    args = parse_args()
    os.makedirs(args.click_save_folder, exist_ok=True)
    image_folder = os.path.join(args.image_folder, "imgs")
    with open(args.color_map_path, "r") as f:
        color_map = json.load(f)
        color_map = {tuple(map(int, k.strip("()").split(","))): v for k, v in color_map.items()}
        mc_color = [k for k, v in color_map.items() if v == args.target_class][0]

    og = omniglue.OmniGlue(
        og_export="./checkpoints/omniglue.onnx",
        sp_export="./checkpoints/sp_v6.onnx",
        dino_export="./checkpoints/dinov2_vitb14_pretrain.pth",
    )

    with open(args.index_file, "r") as f:
        frame_list = [line.strip() for line in f if line.strip().endswith(".jpg")]

    for fname in frame_list:
        stem = os.path.splitext(fname)[0]
        query_path = os.path.join(image_folder,  fname)
        if not os.path.exists(query_path):
            print(f"❌ 缺失帧：{fname},{query_path}")
            continue

        best_support_img = select_best_support(query_path, args.support_label_dir, args.support_img_dir, args.target_class)
        support_mask_path = os.path.join(args.support_mask_dir, os.path.basename(best_support_img).replace(".jpg", "_mask.png"))

        mask_img = Image.open(support_mask_path).convert("RGB")
        mask_np = np.array(mask_img)
        mc_mask = np.all(mask_np == mc_color, axis=-1)

        image0 = np.array(Image.open(query_path).convert("RGB"))
        image1 = np.array(Image.open(best_support_img).convert("RGB"))
        match_kp0, match_kp1, confidences = og.FindMatches(image0, image1)

        keep = confidences > 0.02
        match_kp0 = match_kp0[keep]
        match_kp1 = match_kp1[keep]

        labels = []
        for pt0, pt1 in zip(match_kp0, match_kp1):
            x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
            if 0 <= y1 < mc_mask.shape[0] and 0 <= x1 < mc_mask.shape[1]:
                region = mc_mask[max(0, y1 - 5):y1 + 5, max(0, x1 - 5):x1 + 5]
                label = 1 if np.any(region) else 0
            else:
                label = 0
            labels.append((pt0[0], pt0[1], label))

        patch_features = extract_dinov2_feature(query_path, fit_pca=True)  # (196, 32)
        patch_h, patch_w = 14, 14
        img_h, img_w = 224, 224
        patch_size_h = img_h // patch_h
        patch_size_w = img_w // patch_w
        match_kpt_file = os.path.join(args.match_kpt_folder, f"{stem}.txt")

        # 获取 match_kp0 中 label=1 的点的 patch feature

        # 获取 match_kp0 中 label=1 的点的 patch feature（降维后）
        positive_features = []
        for x, y, label in labels:
            if label == 1:
                px = int(x / image0.shape[1] * 224) // patch_size_w
                py = int(y / image0.shape[0] * 224) // patch_size_h
                idx = py * patch_w + px
                if 0 <= idx < len(patch_features):
                    positive_features.append(patch_features[idx])

        match_kpt_file = os.path.join(args.match_kpt_folder, f"{stem}.txt")
        if os.path.exists(match_kpt_file) and positive_features:
            with open(match_kpt_file, "r") as f:
                for line in f:
                    x, y = map(float, line.strip().split(",")[:2])
                    px = int(x / image0.shape[1] * 224) // patch_size_w
                    py = int(y / image0.shape[0] * 224) // patch_size_h
                    idx = py * patch_w + px
                    if 0 <= idx < len(patch_features):
                        query_vec = patch_features[idx]
                        sims = [cosine_similarity(query_vec, pos_vec) for pos_vec in positive_features]
                        label = 1 if max(sims) > 0.9 else 0
                        #labels.append((x, y, label))


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
