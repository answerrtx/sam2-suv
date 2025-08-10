import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import requests
from io import BytesIO

# 使用 Meta 官方模型
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2.eval()
path0="./Datasets/SPUS/DIFF/S1602/imgs/00024.jpg"
path0="./Datasets/MSKUSO/view_feature/hp8/imgs/00035.jpg"
vis = path0.replace('imgs', 'vis').replace('.jpg', '_vis.png')
#path0="./Datasets/MSKUSO/hp12/imgs/00029.jpg"

#path0="./Datasets/MSKUSO/support/hip_imgs/hp_00004.jpg"

#path0="./Datasets/MSKUSO/shoulder/shd9/00029.jpg"

# 加载图像（你可以改成自己的路径）
image = Image.open(path0).convert("RGB")
vis = Image.open(vis).convert("RGB")
# DINOv2 标准预处理
transform = transforms.Compose([
    transforms.Resize((518, 518)),  # ViT patch size 通常是 14x14 or 16x16
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
img_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
with torch.no_grad():
    output = dinov2.forward_features(img_tensor)
    features = output['x_norm_patchtokens']  # shape: [1, num_patches, feat_dim]
    features = features.squeeze(0)  # shape: [num_patches, feat_dim]

# t-SNE 降维到 2D
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
features_tsne = tsne.fit_transform(features.cpu().numpy())

# 计算 patch 网格尺寸
num_patches = features_tsne.shape[0]
grid_size = int(np.sqrt(num_patches))  # e.g. 37
assert grid_size * grid_size == num_patches, "Not a square grid"

# 将 t-SNE 值归一化
features_tsne = tsne.fit_transform(features.cpu().numpy())
tsne_grid = features_tsne.reshape(grid_size, grid_size, 2)  # ✅ 行优先 reshape

tsne_min = tsne_grid.min(axis=(0,1), keepdims=True)
tsne_max = tsne_grid.max(axis=(0,1), keepdims=True)
tsne_norm = (tsne_grid - tsne_min) / (tsne_max - tsne_min)  # [37, 37, 2]

tsne_rgb = np.concatenate([
    tsne_norm, 
    1 - np.linalg.norm(tsne_norm - 0.5, axis=2, keepdims=True)
], axis=2)  # [37, 37, 3]

# 将 t-SNE 映射归一化后的结果转换为图像
tsne_uint8 = (tsne_rgb * 255).astype(np.uint8)           # [37, 37, 3]
tsne_img_pil = Image.fromarray(tsne_uint8)               # 转成 PIL 图像

# 用 NEAREST 插值 resize 到原图大小，保持块状结构清晰
tsne_img_resized = tsne_img_pil.resize(image.size, resample=Image.NEAREST)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(tsne_img_resized)
axs[0].set_title("t-SNE Patch Grid (Resized, NEAREST)")
axs[0].axis('off')

axs[1].imshow(image)
axs[1].set_title("Original Image")
axs[1].axis('off')

plt.tight_layout()
plt.show()
