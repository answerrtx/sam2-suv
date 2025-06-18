import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import numpy as np

# 使用 Meta 官方模型
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2.eval()

# 图像路径设置
path0 = "./Datasets/MSKUSO/view_feature/S1602/imgs/00035.jpg"
vis = path0.replace('imgs', 'vis').replace('.jpg', '_vis.png')

image = Image.open(path0).convert("RGB")
vis = Image.open(vis).convert("RGB")

# 标准预处理
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
img_tensor = transform(image).unsqueeze(0)

# 提取 DINOv2 特征
with torch.no_grad():
    output = dinov2.forward_features(img_tensor)
    features = output['x_norm_patchtokens'].squeeze(0).cpu().numpy()

# 使用 UMAP 降维到 2D
print("Running UMAP...")
umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
features_umap = umap_model.fit_transform(features)  # shape: [num_patches, 2]

# 计算 patch 网格尺寸（例如 37x37）
num_patches = features_umap.shape[0]
grid_size = int(np.sqrt(num_patches))
assert grid_size * grid_size == num_patches, "Not a square grid"

# 重塑为网格结构
umap_grid = features_umap.reshape(grid_size, grid_size, 2)

# 归一化
umap_min = umap_grid.min(axis=(0, 1), keepdims=True)
umap_max = umap_grid.max(axis=(0, 1), keepdims=True)
umap_norm = (umap_grid - umap_min) / (umap_max - umap_min)

# 映射为 RGB
umap_rgb = np.concatenate([
    umap_norm,
    1 - np.linalg.norm(umap_norm - 0.5, axis=2, keepdims=True)
], axis=2)

# 转换为图像并放大
umap_uint8 = (umap_rgb * 255).astype(np.uint8)
umap_img_pil = Image.fromarray(umap_uint8)
umap_img_resized = umap_img_pil.resize(image.size, resample=Image.NEAREST)

# 可视化：UMAP vs 原图
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(umap_img_resized)
axs[0].set_title("UMAP Patch Grid (Resized, NEAREST)")
axs[0].axis('off')

axs[1].imshow(image)
axs[1].set_title("Original Image")
axs[1].axis('off')

plt.tight_layout()
plt.show()
