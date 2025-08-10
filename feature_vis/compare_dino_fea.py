import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# 加载 DINOv2 模型
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2.eval()

# 图像路径
path0 = "./Datasets/SPUS/DIFF/S1602/imgs/00024.jpg"
path1 = "./Datasets/SPUS/DIFF/S1602/imgs/00020.jpg"

#path1 = "./Datasets/MSKUSO/hp12/imgs/00029.jpg"

# 标准预处理
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = dinov2.forward_features(img_tensor)
        features = output['x_norm_patchtokens'].squeeze(0)
    return features

# MMD 计算函数
def compute_linear_mmd(x, y):
    return torch.norm(x.mean(dim=0) - y.mean(dim=0), p=2)


# 提取特征
features0 = extract_features(path0)
features1 = extract_features(path1)

# 计算 cosine 相似度
cos_sim = F.cosine_similarity(features0, features1, dim=1)
num_patches = features0.shape[0]
grid_size = int(np.sqrt(num_patches))
assert grid_size * grid_size == num_patches, "Not a square patch grid."
cos_sim_grid = cos_sim.reshape(grid_size, grid_size).cpu().numpy()

# 可视化 Cosine 相似度
plt.figure(figsize=(6, 6))
plt.imshow(cos_sim_grid, cmap='viridis')
plt.colorbar(label='Cosine Similarity')
plt.title("Patch-wise Cosine Similarity Between Two Images")
plt.axis('off')
plt.tight_layout()
plt.show()

# 计算并打印 MMD 值
mmd_val = compute_linear_mmd(features0, features1).item()
cos_sim = F.cosine_similarity(
    features0.mean(dim=0, keepdim=True),
    features1.mean(dim=0, keepdim=True),
    dim=1
).item()

print("L2 Distance:", torch.norm(features0 - features1).item())
print("Linear MMD:", compute_linear_mmd(features0, features1).item())
print("Mean Cosine:", F.cosine_similarity(features0.mean(0, keepdim=True), features1.mean(0, keepdim=True), dim=1).item())

# 简单判断
is_cross = (cos_sim < 0.995 or mmd_val > 2.0 or torch.norm(features0 - features1).item() > 600)
print("🧠 Cross-Domain:", "✅ YES" if is_cross else "❌ NO")