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
from transformers import AutoImageProcessor, Dinov2Model

def load_label_map(suv_json_path):
    with open(suv_json_path, 'r') as f:
        color_map = json.load(f)
    return {eval(k): v for k, v in color_map.items()}

def get_color(label):
    np.random.seed(hash(label) % (2**32))
    return np.random.rand(3,)

def extract_features(img, model, processor, device):
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    feats = outputs.last_hidden_state[0, 1:, :]  # (N_patches, dim)
    return feats.cpu()

def map_coords_to_patch(coords, img_size=(224, 224), patch_grid=(14, 14)):
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
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    label_map = load_label_map(suv_json)

    features = []
    labels = []

    subdirs = sorted([d for d in os.listdir(root_dir) if not d.endswith('_mask')])
    for sub in tqdm(subdirs):
        img_dir = os.path.join(root_dir, sub)
        mask_dir = os.path.join(root_dir, sub + '_mask')
        if not os.path.exists(mask_dir):
            continue

        img_list = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        if len(img_list) == 0:
            continue

        selected_imgs = random.sample(img_list, min(max_per_folder, len(img_list)))
        print(selected_imgs)
        for img_name in selected_imgs:
            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name.replace('.jpg', '_mask.png').replace('.jpeg', '.png'))
            if not os.path.exists(mask_path):
                continue

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("RGB")
            img_resized = preprocess(img)
            print(img.size, mask.size)
            mask_resized = mask.resize((224, 224), resample=Image.NEAREST)

            feat = extract_features(img, model, processor, device)  # [196, 768]
            mask_np = np.array(mask_resized)

            for color, label in label_map.items():
                match = np.all(mask_np == color, axis=-1)
                ys, xs = np.where(match)
                coords = list(zip(xs, ys))
                if not coords:
                    continue
                patch_ids = map_coords_to_patch(coords, img_size=(224, 224))
                selected_feats = feat[patch_ids]
                mean_feat = selected_feats.mean(dim=0).numpy()
                features.append(mean_feat)
                labels.append(label)

    return np.array(features), labels

def visualize(features, labels, out_path="tsne_dino_features.png"):
    label_set = sorted(set(labels))
    label_to_color = {l: get_color(l) for l in label_set}

    print("Running PCA and t-SNE...")
    pca = PCA(n_components=50).fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(features)

    plt.figure(figsize=(10, 8))
    for label in label_set:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(tsne[indices, 0], tsne[indices, 1], c=[label_to_color[label]], label=label, alpha=0.6)

    plt.title("t-SNE of DINOv2 Features by Class (random sampled folders)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True, help='Path to MSKUSO dataset folder')
    parser.add_argument('--suv_json', required=True, help='Path to suv.json mapping')
    parser.add_argument('--max_per_folder', type=int, default=3, help='How many images per subfolder to sample')
    args = parser.parse_args()

    feats, labels = process_dataset(args.root_dir, args.suv_json, args.max_per_folder)
    visualize(feats, labels)
