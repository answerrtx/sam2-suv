import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# 配置路径
input_root = "results/folder"
output_folder = "results/merged_folder"
os.makedirs(output_folder, exist_ok=True)

# 加载 SUV 标签颜色映射
with open("suv.json", "r") as f:
    suv_data = json.load(f)
color_map_raw = suv_data.get("color_map", {})
color_map = {v: eval(k) for k, v in color_map_raw.items()}  # e.g., {"bone": (255, 255, 255)}

# 获取所有 mask_folder
mask_folders = [os.path.join(input_root, d) for d in os.listdir(input_root)
                if os.path.isdir(os.path.join(input_root, d))]

# 假设所有 mask 大小一致，用第一张 mask 尺寸做初始化
sample_file = next(os.walk(mask_folders[0]))[2][0]
sample_path = os.path.join(mask_folders[0], sample_file)
height, width = np.array(Image.open(sample_path)).shape[:2]

# 遍历所有图片名
all_filenames = sorted(os.listdir(mask_folders[0]))  # 假设每个子文件夹都包含相同文件名

for filename in tqdm(all_filenames, desc="Merging masks"):
    merged_mask = np.zeros((height, width, 3), dtype=np.uint8)  # RGB 合并图像初始化

    for folder in mask_folders:
        mask_path = os.path.join(folder, filename)
        if not os.path.exists(mask_path):
            continue
        mask_img = np.array(Image.open(mask_path))

        # 如果是灰度图，直接用 label index 或者名称转颜色
        unique_labels = np.unique(mask_img)
        for label in unique_labels:
            if label == 0:  # 忽略背景
                continue

            label_mask = mask_img == label
            label_str = str(label) if str(label) in color_map else int(label)
            color = color_map.get(label_str, (128, 128, 128))  # 默认灰色
            merged_mask[label_mask] = color

    # 保存合并图像
    merged_img = Image.fromarray(merged_mask)
    merged_img.save(os.path.join(output_folder, filename))

print(f"✅ 合并完成，结果保存在：{output_folder}")
