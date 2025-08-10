import os
import json
from PIL import Image

def load_color_map_flat(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return set(eval(k) for k in data.keys())  # 直接取顶层键

def find_unknown_colors(image_path, valid_colors):
    img = Image.open(image_path).convert("RGB")
    pixels = img.getdata()
    unknown_colors = set(pixels) - valid_colors
    return unknown_colors

def check_folder_for_unknown_colors(image_folder, json_path):
    valid_colors = load_color_map_flat(json_path)
    print(f"Loaded {len(valid_colors)} valid colors from {json_path}.")

    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg")):
                img_path = os.path.join(root, file)
                unknown = find_unknown_colors(img_path, valid_colors)
                if unknown:
                    print(f"[!] Unknown colors in {img_path}:")

if __name__ == "__main__":
    # 修改为你自己的路径
    folder_path = "./Datasets/MSKUSO/support/mask"
    suv_json_path = "./suv.json"
    
    check_folder_for_unknown_colors(folder_path, suv_json_path)
