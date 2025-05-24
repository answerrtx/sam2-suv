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
path0="./Datasets/SPUS/S1205/00024.jpg"
path0="./Datasets/SPUS/S1702/00024.jpg"

#path0="./Datasets/MSKUSO/support/hip_imgs/hp_00004.jpg"

#path0="./Datasets/MSKUSO/shoulder/shd9/00029.jpg"

# 加载图像（你可以改成自己的路径）
image = Image.open(path0).convert("RGB")

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
tsne_norm = (features_tsne - features_tsne.min(0)) / (features_tsne.max(0) - features_tsne.min(0))

# 创建一个 RGB 图像（将 2D t-SNE 映射成 3D）
tsne_rgb = np.zeros((grid_size, grid_size, 3))
tsne_rgb[..., 0:2] = tsne_norm.reshape(grid_size, grid_size, 2)  # t-SNE 两维映射到 R/G
tsne_rgb[..., 2] = 1 - np.linalg.norm(tsne_norm - 0.5, axis=1).reshape(grid_size, grid_size)  # 用距离中心的强度作为 B 通道

# 可视化 Patch 特征热图
plt.figure(figsize=(6, 6))
plt.imshow(tsne_rgb)
plt.title("t-SNE Projection Mapped to Patch Grid")
plt.axis('off')
plt.tight_layout()
plt.show()