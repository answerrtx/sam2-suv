import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import requests
from io import BytesIO

# 使用 Meta 官方模型
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2.eval()

#path0="./Datasets/MSKUSO/hnd3/00035.jpg"
path0="./Datasets/SPUS/S1205/00024.jpg"
path0="./Datasets/MSKUSO/shd14/imgs/00020.jpg"
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
# PCA降维到3维用于RGB可视化
pca = PCA(n_components=3)
features_rgb = pca.fit_transform(features.cpu().numpy())

# 归一化到 [0, 1]
features_rgb -= features_rgb.min()
features_rgb /= features_rgb.max()

# 还原为原来的 patch grid（以 ViT-14 为例：518/14 = 37）
grid_size = int(features_rgb.shape[0] ** 0.5)
img_feat = features_rgb.reshape(grid_size, grid_size, 3)

plt.imshow(img_feat)
plt.title("DINOv2 Patch Feature Visualization (PCA)")
plt.axis('off')
plt.show()
