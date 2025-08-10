"""
非常单纯地拿到指定点的image feature
"""

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


# 输入点（原图像坐标）
points = [(100, 200), (300, 400)]  # 替换为你自己的点

# 将点映射到 patch grid
patch_size = 14
grid_size = 518 // patch_size  # 37


print("==== Embeddings for each point ====")
for idx, (x, y) in enumerate(points):
    px, py = x // patch_size, y // patch_size
    patch_index = py * grid_size + px  # row-major
    embedding = patch_features[patch_index].cpu().numpy()
    
    print(f"\nPoint {idx+1} at (x={x}, y={y}) mapped to patch ({px}, {py}) index {patch_index}")
    print("Embedding vector:")
    print(embedding,len(embedding))