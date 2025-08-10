import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from tqdm import tqdm
import random
# 路径配置
img_folder = "./Datasets/MSKUSO/hnd3/imgs"
mask_folder = "./Datasets/MSKUSO/hnd3/mask"
suv_json_path = "./suv_all.json"

# 加载 SUV coarse-fine 标签和颜色映射
with open(suv_json_path, 'r') as f:
    suv_data = json.load(f)
color_map = {eval(k): v for k, v in suv_data['color_map'].items()}
fine_to_coarse = {
    fine: coarse
    for coarse, fine_list in suv_data.items()
    if coarse != 'color_map'
    for fine in fine_list
}

# 加载 DINOv2 模型
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# 遍历文件夹中所有图像，提取特征
all_features = []
all_labels_fine = []
all_labels_coarse = []

image_files = sorted([f for f in os.listdir(img_folder) if f.lower().endswith(".jpg")])

if len(image_files) == 0:
    print("No valid image-mask pairs found.")
    

sampled_pairs = random.sample(image_files, min(50, len(image_files)))


for fname in tqdm(sampled_pairs, desc="Processing images"):
    try:
        img_path = os.path.join(img_folder, fname)
        mask_path = os.path.join(mask_folder, fname.replace(".jpg", "_mask.png"))
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # 图像和 mask 预处理
        img_tensor = transform(image).unsqueeze(0)
        mask_resized = mask.resize((14, 14), resample=Image.NEAREST)
        mask_np = np.array(mask_resized)

        # 提取 patch 特征
        with torch.no_grad():
            output = dinov2.forward_features(img_tensor)
            features = output['x_norm_patchtokens'].squeeze(0)  # [num_patches, dim]

        # 遍历 patch grid
        for y in range(37):
            for x in range(37):
                idx = y * 37 + x
                color = tuple(mask_np[y, x])
                if color in color_map:
                    fine_label = color_map[color]
                    coarse_label = fine_to_coarse.get(fine_label, 'Unknown')
                    all_features.append(features[idx].numpy())
                    all_labels_fine.append(fine_label)
                    all_labels_coarse.append(coarse_label)
    except Exception as e:
        print(f"Error processing {fname}: {e}")

if len(all_features) == 0:
    raise ValueError("No valid labels found in the dataset.")

print("Applying PCA to reduce from 768 -> 50")
pca = PCA(n_components=50, random_state=42)
features_pca = pca.fit_transform(np.array(all_features))


# 降维
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_2d = tsne.fit_transform(np.array(all_features))

# 为每个标签分配颜色
def get_color_map(labels, cmap_name='tab20'):
    unique_labels = sorted(set(labels))
    cmap = cm.get_cmap(cmap_name, len(unique_labels))
    return {label: cmap(i) for i, label in enumerate(unique_labels)}, unique_labels

fine_color_map, fine_unique = get_color_map(all_labels_fine)
coarse_color_map, coarse_unique = get_color_map(all_labels_coarse)

# 并排可视化
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 细粒度图
for label in fine_unique:
    idxs = [i for i, l in enumerate(all_labels_fine) if l == label]
    coords = features_2d[idxs]
    axs[0].scatter(coords[:, 0], coords[:, 1], label=label, color=fine_color_map[label], s=10, alpha=0.7)
axs[0].set_title("Fine-Grained Labels")
axs[0].set_xlabel("t-SNE Dim 1")
axs[0].set_ylabel("t-SNE Dim 2")
axs[0].legend(fontsize='x-small', bbox_to_anchor=(1.05, 1), loc='upper left')

# 粗粒度图
for label in coarse_unique:
    idxs = [i for i, l in enumerate(all_labels_coarse) if l == label]
    coords = features_2d[idxs]
    axs[1].scatter(coords[:, 0], coords[:, 1], label=label, color=coarse_color_map[label], s=10, alpha=0.7)
axs[1].set_title("Coarse-Grained Labels")
axs[1].set_xlabel("t-SNE Dim 1")
axs[1].set_ylabel("t-SNE Dim 2")
axs[1].legend(fontsize='x-small', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
