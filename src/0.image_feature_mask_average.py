import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoImageProcessor, Dinov2Model
from collections import defaultdict

import json
# 初始化全局存储
global_embeddings_sum = defaultdict(lambda: np.zeros(768, dtype=np.float32))
global_embeddings_count = defaultdict(int)
# 初始化 DINOv2 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = Dinov2Model.from_pretrained(model_name).to(device)
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # DINOv2 默认输入
    transforms.ToTensor()
])

# 加载 mask → 类别映射
color_map = {
    (255, 0, 0): "VA",
    (0, 255, 0): "MC",
    (0, 0, 255): "PrxPX",
    (255, 255, 0): "RCL",
    (255, 49, 0): "ADd",
    (255, 98, 0): "UCL",
    (255, 148, 0): "FT",
    (255, 197, 0): "Sca",
    (255, 246, 0): "LU",
    (213, 255, 0): "T",
    (164, 255, 0): "Rad",
    (115, 255, 0): "bone",
    (255, 99, 71): "FH",
    (135, 206, 250): "FN",
    (124, 252, 0): "IP",
    (255, 165, 0): "RF",
    (138, 43, 226): "Cap",
    (255, 20, 147): "AIIS",
    (64, 224, 208): "Subscap",
    (255, 215, 0): "SSP",
    (75, 0, 130): "Deltoid",
    (34, 139, 34): "Hum",
    (70, 130, 180): "ACR",
    (240, 128, 128): "CLV",
    (199, 21, 133): "InfSp",
    (0, 191, 255): "P",
    (255, 140, 0): "MCL",
    (147, 112, 219): "Fem",
    (60, 179, 113): "Tib",
    (220, 20, 60): "Vessel",
    (255, 105, 180): "Biceps",
    (0, 128, 128): "A",
    (255, 69, 0): "GtrTroch",
    (106, 90, 205): "VB",
    (0, 255, 255): "S",
    (255, 0, 255): "TP",
    (173, 255, 47): "AP",
    (139, 0, 139): "L",
    (255, 228, 181): "Ul",
    (0, 250, 154): "EDM",
    (173, 216, 230): "ED",
    (152, 251, 152): "Ext",
    (205, 92, 92): "Label_43",
    (72, 209, 204): "Label_44",
    (0, 0, 139): "Label_45",
    (210, 105, 30): "Label_46",
    (255, 160, 122): "Label_47",
    (123, 104, 238): "Label_48",
    (46, 139, 87): "Label_49",
    (176, 224, 230): "Label_50",
    (0, 0, 0): "__bg__"
}


def get_mask_regions(mask_np):
    regions = {}
    for color, label in color_map.items():
        region = np.all(mask_np == color, axis=-1)
        coords = np.argwhere(region)
        if coords.shape[0] > 0:
            regions[label] = coords
    return regions

def extract_dino_features(img_tensor):
    inputs = processor(images=img_tensor, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state  # shape: (1, 197, 768)
    return features[0, 1:, :]  # 去掉 CLS token → shape: (196, 768)

def map_coords_to_patch(coords, orig_size, patch_size=14):
    # 把原图坐标映射到 14x14 patch grid 的索引
    h, w = orig_size
    coords = coords.astype(np.float32)
    grid_y = np.floor(coords[:, 0] / h * 14).astype(int)
    grid_x = np.floor(coords[:, 1] / w * 14).astype(int)
    patch_idx = grid_y * 14 + grid_x
    patch_idx = np.clip(patch_idx, 0, 195)
    return patch_idx

def process_one(image_path, mask_path):
    #print(image_path,"!")
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("RGB")

    image_tensor = preprocess(image)
    features = extract_dino_features(image_tensor)

    mask_np = np.array(mask)
    regions = get_mask_regions(mask_np)
    avg_embeddings = {}

    for label, coords in regions.items():
        #print(image.size)
        patch_ids = map_coords_to_patch(coords, orig_size=image.size[::-1])  # PIL: (W,H)
        selected_features = features[patch_ids]
        avg_feat = selected_features.mean(dim=0).cpu().numpy()
        avg_embeddings[label] = avg_feat
    #print(avg_embeddings)
    return avg_embeddings

# 主流程
"""
#for 画图
#可视化代码去看看这个文件夹里的不同label的feature的区分程度在 feature_distribution.py
image_dir = "./Datasets/MSKUSO/shoulder/shd2"
mask_dir = "./Datasets/MSKUSO/shoulder/shd2_mask"
output_dir = "./Datasets/MSKUSO/shoulder/shd2_features"
"""
image_dir = "./Datasets/MSKUSO/support/hand_wrist_imgs"
mask_dir = "./Datasets/MSKUSO/support/hand_wrist_mask"
output_dir = "./Datasets/MSKUSO/support/hand_wrist_mask_features"
os.makedirs(output_dir, exist_ok=True)

for fname in tqdm(os.listdir(image_dir)):
    if not fname.endswith(".jpg"):
        continue
    if image_dir.find("support") != -1:
        mask_path = os.path.join(mask_dir, fname.replace(".jpg", "_mask.png"))
    else:
        mask_path = os.path.join(mask_dir, fname.replace(".jpg", "_mask.png"))
    image_path = os.path.join(image_dir, fname)
    #if not os.path.exists(image_path):
    #    continue
    features = process_one(image_path, mask_path)
    # 保存每张图中各类别的平均特征
    features_serializable = {label: feat.tolist() for label, feat in features.items()}
    out_path = os.path.join(output_dir, f"{fname}_avg_features.json").replace(".jpg", "")
    with open(out_path, 'w') as f:
        json.dump(features_serializable, f, indent=2)

    for label, feat in features.items():
        global_embeddings_sum[label] += feat
        global_embeddings_count[label] += 1

# 最终计算平均
global_avg_embeddings = {
    label: (global_embeddings_sum[label] / global_embeddings_count[label]).tolist()
    for label in global_embeddings_sum
}
# 保存到 json
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "mask_avg_features.json")
with open(output_path, 'w') as f:
    json.dump(global_avg_embeddings, f, indent=2)
