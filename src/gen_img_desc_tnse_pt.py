import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as patches

# 加载 DINOv2 模型
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2.eval()

# 加载图像并预处理
path0="./Datasets/MSKUSO/shoulder/shd9/00029.jpg"


image = Image.open(path0).convert("RGB")
resized_image = image.resize((518, 518))  # 固定大小以适配 ViT patch grid

transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
img_tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 518, 518]

# 提取 patch-level 特征
with torch.no_grad():
    output = dinov2.forward_features(img_tensor)
    patch_features = output['x_norm_patchtokens'].squeeze(0)  # [num_patches, dim]




# t-SNE 降维
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
tsne_result = tsne.fit_transform(patch_features.cpu().numpy())  # [1369, 2]

# 映射回 patch grid
grid_size = int(np.sqrt(tsne_result.shape[0]))  # 应该是 37
tsne_grid = tsne_result.reshape(grid_size, grid_size, 2)

# 归一化为 RGB 可视化
tsne_min = tsne_result.min(axis=0)
tsne_max = tsne_result.max(axis=0)
tsne_norm = (tsne_result - tsne_min) / (tsne_max - tsne_min)
tsne_rgb = tsne_norm.reshape(grid_size, grid_size, 2)
tsne_rgb = np.concatenate([tsne_rgb, 1 - np.linalg.norm(tsne_rgb - 0.5, axis=2, keepdims=True)], axis=2)

# 输入点（原图像坐标）
points = [(100, 200), (300, 400)]  # 替换为你自己的点

# 将点映射到 patch grid
patch_size = 14
patch_coords = [(x // patch_size, y // patch_size) for (x, y) in points]

# 创建双图显示
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 1. 原图 + 红点标记
ax1.imshow(resized_image)
for x, y in points:
    ax1.plot(x, y, 'ro', markersize=6)
ax1.set_title("Original Image with Points")
ax1.axis('off')

# 2. t-SNE 热图 + 红框标记对应 patch
ax2.imshow(tsne_rgb)
for px, py in patch_coords:
    rect = patches.Rectangle((px-0.5, py-0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
    ax2.add_patch(rect)
ax2.set_title("t-SNE Patch Grid Visualization")
ax2.axis('off')


plt.tight_layout()
plt.show()

print("==== Embeddings for each point ====")
for idx, (x, y) in enumerate(points):
    px, py = x // patch_size, y // patch_size
    patch_index = py * grid_size + px  # row-major
    embedding = patch_features[patch_index].cpu().numpy()
    
    print(f"\nPoint {idx+1} at (x={x}, y={y}) mapped to patch ({px}, {py}) index {patch_index}")
    print("Embedding vector:")
    print(embedding,len(embedding))