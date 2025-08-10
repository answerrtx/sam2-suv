import os
import json
import argparse
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import transforms
import matplotlib.cm as cm


def get_color_map(label_list, cmap_name='tab20'):
    cmap = cm.get_cmap(cmap_name, len(label_list))
    return {label: cmap(i) for i, label in enumerate(label_list)}


def load_label_map(suv_json_path):
    with open(suv_json_path, 'r') as f:
        suv_data = json.load(f)
    fine_to_coarse = {
        fine: coarse
        for coarse, fine_list in suv_data.items()
        if coarse != 'color_map'
        for fine in fine_list
    }
    color_map = {eval(k): v for k, v in suv_data['color_map'].items()}
    return color_map, fine_to_coarse


def extract_features(img_tensor, model):
    with torch.no_grad():
        outputs = model.forward_features(img_tensor)
    return outputs['x_norm_patchtokens'].squeeze(0).cpu()  # shape: [1376, 768]


def map_coords_to_patch(coords, img_size=(518, 518), patch_grid=(37, 37)):
    H, W = img_size
    ph, pw = patch_grid
    mapped = []
    for x, y in coords:
        px = int(x / W * pw)
        py = int(y / H * ph)
        idx = py * pw + px
        mapped.append(idx)
    return mapped


def process_dataset(root_dir, suv_json, max_per_folder=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    color_map, fine_to_coarse = load_label_map(suv_json)

    features_fine, labels_fine = [], []
    features_coarse, labels_coarse = [], []

    subdirs = sorted(os.listdir(root_dir))
    for sub in tqdm(subdirs):
        img_dir = os.path.join(root_dir, sub, "imgs")
        mask_dir = os.path.join(root_dir, sub, "mask")
        if '.json' in sub or 'support' in sub or not os.path.exists(mask_dir):
            continue

        img_list = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        if not img_list:
            continue

        selected_imgs = random.sample(img_list, min(max_per_folder, len(img_list)))
        for img_name in selected_imgs:
            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name.replace('.jpg', '_mask.png').replace('.jpeg', '.png'))
            if not os.path.exists(mask_path):
                print("Missing mask:", mask_path)
                continue

            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
            mask_resized = mask.resize((518, 518), resample=Image.NEAREST)
            mask_np = np.array(mask_resized)

            feats = extract_features(image_tensor, model)  # shape: [1376, 768]

            for color, fine_label in color_map.items():
                coarse_label = fine_to_coarse.get(fine_label, "Unknown")
                if(coarse_label=="Unknown"): continue
                match = np.all(mask_np == color, axis=-1)
                ys, xs = np.where(match)
                coords = list(zip(xs, ys))
                if not coords:
                    continue

                patch_ids = map_coords_to_patch(coords, img_size=(518, 518), patch_grid=(37, 37))
                selected_feats = feats[patch_ids]
                mean_feat = selected_feats.mean(dim=0).numpy()

                features_fine.append(mean_feat)
                labels_fine.append(fine_label)
                features_coarse.append(mean_feat)
                labels_coarse.append(coarse_label)

    return (
        np.array(features_fine), labels_fine,
        np.array(features_coarse), labels_coarse
    )


def visualize(features1, labels1, features2, labels2, out_path="tsne_dual_dino_features.png"):
    pca1 = PCA(n_components=50).fit_transform(features1)
    tsne1 = TSNE(n_components=2, perplexity=30, init='pca', random_state=42).fit_transform(features1)

    pca2 = PCA(n_components=50).fit_transform(features2)
    tsne2 = TSNE(n_components=2, perplexity=30, init='pca', random_state=42).fit_transform(features2)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    label_set1 = sorted(set(labels1))
    color_map1 = get_color_map(label_set1, cmap_name='tab20')
    for label in label_set1:
        indices = [i for i, l in enumerate(labels1) if l == label]
        axes[0].scatter(tsne1[indices, 0], tsne1[indices, 1],
                        c=[color_map1[label]], label=label, alpha=1, s=20)
    axes[0].set_title("t-SNE of DINOv2 Features by Fine-grained Labels")
    #axes[0].legend(fontsize='x-small', markerscale=1, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].legend(fontsize='large', markerscale=1, loc='best')
    

    label_set2 = sorted(set(labels2))
    color_map2 = get_color_map(label_set2, cmap_name='tab10')
    for label in label_set2:
        indices = [i for i, l in enumerate(labels2) if l == label]
        axes[1].scatter(tsne2[indices, 0], tsne2[indices, 1],
                        c=[color_map2[label]], label=label, alpha=1, s=20)
    axes[1].set_title("t-SNE of DINOv2 Features by Coarse-grained Labels")
    #axes[1].legend(fontsize='x-small', markerscale=1, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].legend(fontsize='large', markerscale=1, loc='best')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True, help='Path to MSKUSO dataset folder')
    parser.add_argument('--suv_json', required=True, help='Path to suv.json mapping')
    parser.add_argument('--max_per_folder', type=int, default=5, help='How many images per subfolder to sample')
    parser.add_argument('--out_path', type=str, default='tsne_dual_dino_features.png', help='Output path for visualization')
    args = parser.parse_args()

    feats_fine, labels_fine, feats_coarse, labels_coarse = process_dataset(
        args.root_dir, args.suv_json, args.max_per_folder
    )
    visualize(feats_fine, labels_fine, feats_coarse, labels_coarse, out_path=args.out_path)
