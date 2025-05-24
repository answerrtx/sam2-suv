import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.cm as cm

# 路径配置

image_path = "./Datasets/MSKUSO/hp8/imgs/00035.jpg"
mask_path = "./Datasets/MSKUSO/hp8/mask/00035_mask.png"
#image_path = "./Datasets/SPUS/DIFF/S1602/imgs/00035.jpg"
#mask_path = "./Datasets/SPUS/DIFF/S1602/mask/00035_mask.png"
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
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
image = Image.open(image_path).convert("RGB")
img_tensor = transform(image).unsqueeze(0)

# 提取 patch-level feature
with torch.no_grad():
    output = dinov2.forward_features(img_tensor)
    features = output['x_norm_patchtokens'].squeeze(0)  # [num_patches, dim]

# 读取 mask 并resize到 patch grid大小（518/14≈37）
mask = Image.open(mask_path).convert("RGB")
mask_resized = mask.resize((37, 37), resample=Image.NEAREST)
mask_np = np.array(mask_resized)

# 对应每个 patch 提取特征并根据 mask 颜色分类
patch_features = []
patch_labels_fine = []
patch_labels_coarse = []

for y in range(37):
    for x in range(37):
        idx = y * 37 + x
        color = tuple(mask_np[y, x])
        if color in color_map:
            fine_label = color_map[color]
            coarse_label = fine_to_coarse.get(fine_label, 'Unknown')
            patch_features.append(features[idx].numpy())
            patch_labels_fine.append(fine_label)
            patch_labels_coarse.append(coarse_label)

# 如果没有匹配上的点，给出提示
if len(patch_features) == 0:
    raise ValueError("No matching color found in mask file.")

# t-SNE 降维
tsne = TSNE(n_components=2, perplexity=15, random_state=42)
features_2d = tsne.fit_transform(np.array(patch_features))

# 为每个标签分配颜色
def get_color_map(labels, cmap_name='tab20'):
    unique_labels = sorted(set(labels))
    cmap = cm.get_cmap(cmap_name, len(unique_labels))
    return {label: cmap(i) for i, label in enumerate(unique_labels)}, unique_labels

fine_color_map, fine_unique = get_color_map(patch_labels_fine)
coarse_color_map, coarse_unique = get_color_map(patch_labels_coarse)

# 并排绘图
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 细粒度图
for label in fine_unique:
    idxs = [i for i, l in enumerate(patch_labels_fine) if l == label]
    coords = features_2d[idxs]
    axs[0].scatter(coords[:, 0], coords[:, 1], label=label, color=fine_color_map[label], s=20, alpha=0.7)
axs[0].set_title("Fine-Grained Labels")
axs[0].set_xlabel("t-SNE Dim 1")
axs[0].set_ylabel("t-SNE Dim 2")
axs[0].legend(fontsize='x-small', bbox_to_anchor=(1.05, 1), loc='upper left')

# 粗粒度图
for label in coarse_unique:
    idxs = [i for i, l in enumerate(patch_labels_coarse) if l == label]
    coords = features_2d[idxs]
    axs[1].scatter(coords[:, 0], coords[:, 1], label=label, color=coarse_color_map[label], s=20, alpha=0.7)
axs[1].set_title("Coarse-Grained Labels")
axs[1].set_xlabel("t-SNE Dim 1")
axs[1].set_ylabel("t-SNE Dim 2")
axs[1].legend(fontsize='x-small', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
