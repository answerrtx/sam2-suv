import os
import json
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import defaultdict
# ========== 配置 ==========
image_folder = "./Datasets/MSKUSO/support/hand_wrist_imgs"
points_folder = "./Datasets/MSKUSO/support/hand_wrist_rnd_pt"

output_folder = "./Datasets/MSKUSO/support/hand_wrist_pt_features"

patch_size = 14
dinov2_model_name = 'dinov2_vitb14'
image_prefix = ""  # 可选前缀过滤

os.makedirs(output_folder, exist_ok=True)

# ========== 加载 DINOv2 模型 ==========
dinov2 = torch.hub.load('facebookresearch/dinov2', dinov2_model_name)
dinov2.eval()

# ========== 图像预处理 ==========
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

grid_size = 518 // patch_size  # 37
global_feature_pool = defaultdict(list)

# ========== 遍历所有图片 ==========
for fname in tqdm(os.listdir(image_folder)):
    if not (fname.endswith('.jpg') or fname.endswith('.png')):
        continue
    if image_prefix and not fname.startswith(image_prefix):
        continue

    name = os.path.splitext(fname)[0]
    image_path = os.path.join(image_folder, fname)
    json_path = os.path.join(points_folder, f"{name}.json")
    print(json_path)
    if not os.path.exists(json_path):
        print(f"Warning: Missing point file for {fname}")
        continue

    # 读取图像和 DINOv2 特征
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = dinov2.forward_features(img_tensor)
        patch_features = output['x_norm_patchtokens'].squeeze(0)

    # 读取 .txt 格式的标注点（当作 JSON 解析）
    with open(json_path, 'r') as f:
        txt_content = f.read().strip()

    label_points = json.loads(txt_content)
    label_avg = {}

    for label, points in label_points.items():
        embeddings = []
        for (x, y) in points:
            px, py = x // patch_size, y // patch_size
            patch_index = py * grid_size + px
            if patch_index >= patch_features.shape[0]:
                continue
            embedding = patch_features[patch_index].cpu().numpy()
            embeddings.append(embedding)
            global_feature_pool[label].append(embedding)
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            label_avg[label] = avg_embedding.tolist()

    # 保存每张图中各类别的平均特征
    out_path = os.path.join(output_folder, f"{name}_avg_features.json")
    with open(out_path, 'w') as f:
        json.dump(label_avg, f, indent=2)

# ========== 聚合全局平均 ==========
global_avg_features = {}
for label, vectors in global_feature_pool.items():
    avg = np.mean(vectors, axis=0)
    global_avg_features[label] = avg.tolist()

# 保存全局结果
global_out_path = os.path.join(output_folder, "global_avg_features.json")
with open(global_out_path, 'w') as f:
    json.dump(global_avg_features, f, indent=2)
