#!/usr/bin/env python3
# Copyright 2024 Google LLC
# Licensed under the Apache License, Version 2.0

import os
import sys
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

from omniglue_onnx import omniglue

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_mask(mask_path, target_rgb):
    """Load mask and generate a binary mask for the target class."""
    mask_img = np.array(Image.open(mask_path).convert("RGB"))
    binary_mask = np.all(mask_img == np.array(target_rgb), axis=-1)
    return binary_mask
from scipy.ndimage import maximum_filter

def load_mask2(mask_path, target_rgb, tolerance=10):
    mask_img = np.array(Image.open(mask_path).convert("RGB"))
    binary_mask = np.all(mask_img == np.array(target_rgb), axis=-1).astype(np.uint8)

    # 使用 max filter 扩大目标区域
    expanded_mask = maximum_filter(binary_mask, size=(2 * tolerance + 1))
    return expanded_mask.astype(bool)

def load_mask_black_white(mask_path, target_rgb):
    """
    加载彩色 mask 并返回一个黑白二值 mask。
    
    参数:
        mask_path (str): 彩色 mask 文件路径。
        target_rgb (tuple): 目标类别的 RGB 值，例如 (255, 0, 0)。
    
    返回:
        np.ndarray: 黑白二值 mask，目标区域为 255，其余为 0。
    """
    print(mask_path)
    mask_img = np.array(Image.open(mask_path).convert("RGB"))
    binary_mask = np.all(mask_img == np.array(target_rgb), axis=-1).astype(np.uint8) * 255
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_mask, cmap='gray')
    #plt.title(title)
    plt.axis('off')
    plt.show()
    return binary_mask
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image

# 全局变量记录坐标
crop_coords = {}

def line_select_callback(eclick, erelease):
    """
    鼠标释放时触发，记录坐标点
    """
    global crop_coords
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    crop_coords['x1'], crop_coords['x2'] = sorted([x1, x2])
    crop_coords['y1'], crop_coords['y2'] = sorted([y1, y2])
    print(f"📐 选择区域: ({crop_coords['x1']}, {crop_coords['y1']}) -> ({crop_coords['x2']}, {crop_coords['y2']})")

def interactive_crop(image_np):
    fig, ax = plt.subplots(1)
    ax.imshow(image_np)
    ax.set_title("🖱️ 拖动鼠标选择区域，关闭窗口继续")
    toggle_selector = RectangleSelector(
        ax, line_select_callback,
        useblit=True,
        button=[1],  # 左键
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True
    )

    plt.show()  # 阻塞直到关闭窗口

    # 返回裁剪区域
    if crop_coords:
        x1, x2 = crop_coords['x1'], crop_coords['x2']
        y1, y2 = crop_coords['y1'], crop_coords['y2']
        return x1, y1, x2, y2
    else:
        print("⚠️ 没有选择区域")
        return None

def match(image0_fp, image1_fp, mask1_fp, target_class, suv_json_path="./suv.json"):

    # 检查图像是否存在
    for im_fp in [image0_fp, image1_fp, mask1_fp]:
        if not os.path.exists(im_fp) or not os.path.isfile(im_fp):
            raise ValueError(f"Filepath '{im_fp}' doesn't exist or is not a file.")

    # 加载 RGB -> label 映射
    with open(suv_json_path, 'r') as f:
        color_map = json.load(f)
    label_to_rgb = {v: eval(k) for k, v in color_map.items()}
    if target_class not in label_to_rgb:
        raise ValueError(f"Target class '{target_class}' not in suv.json")
    target_rgb = label_to_rgb[target_class]

    print("> Loading images...")
    image0 = np.array(Image.open(image0_fp).convert("RGB"))
    image1 = np.array(Image.open(image1_fp).convert("RGB"))
    
    
    xx1,yy1,xx2,yy2 = interactive_crop(image0)
    cropped = image0[yy1:yy2, xx1:xx2]
    
    ref_mask = load_mask_black_white(mask1_fp, target_rgb)
    print("> Loading OmniGlue...")
    start = time.time()
    og = omniglue.OmniGlue(
        og_export="./checkpoints/omniglue.onnx",
        sp_export="./checkpoints/sp_v6.onnx",
        dino_export="./checkpoints/dinov2_vitb14_pretrain.pth",
    )
    print(f"> \tTook {time.time() - start:.2f} seconds.")

    print("> Finding matches...")
    match_kp0, match_kp1, match_confidences = og.FindMatches(cropped, image1,mask=ref_mask)

    match_threshold = 0.01
    keep_idx = match_confidences > match_threshold
    match_kp0 = match_kp0[keep_idx]
    match_kp1 = match_kp1[keep_idx]

    match_kp0_orig = match_kp0 + np.array([xx1, yy1])

    print(f"> \tFiltered to {len(match_kp0)} matches.")

    # 判断 match_kp1 是否在 mask 区域中
    target_mask = load_mask2(mask1_fp, target_rgb, tolerance=10)
    #in_target = [target_mask[int(y), int(x)] if 0 <= int(y) < target_mask.shape[0] and 0 <= int(x) < target_mask.shape[1] else False for x, y in match_kp1]
    in_target = [
        target_mask[int(y), int(x)] if 0 <= int(y) < target_mask.shape[0] and 0 <= int(x) < target_mask.shape[1]
        else False
        for x, y in match_kp1
    ]
    # 保存 matched 点
    with open("match_res.txt", 'w') as f:
        for pt in match_kp0:
            f.write(f"{pt[0]},{pt[1]}\n")

    print("> Drawing match lines...")
    # 拼接两图
    height = max(image0.shape[0], image1.shape[0])
    viz_img = np.ones((height, image0.shape[1] + image1.shape[1], 3), dtype=np.uint8) * 255
    viz_img[:image0.shape[0], :image0.shape[1]] = image0
    viz_img[:image1.shape[0], image0.shape[1]:] = image1

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(viz_img)
    ax.axis('off')
    from matplotlib.patches import Rectangle

    # 🔵 添加裁剪区域半透明框
    crop_rect = Rectangle(
        (xx1, yy1),  # 左上角
        xx2 - xx1, yy2 - yy1,  # 宽、高
        linewidth=2,
        edgecolor='cyan',
        facecolor='cyan',
        alpha=0.3
    )
    ax.add_patch(crop_rect)

    # 🔴 画匹配点与连线
    for (x0, y0), (x1p, y1p), is_in in zip(match_kp0_orig, match_kp1, in_target):
        x1p += image0.shape[1]  # 拼接图中的右图偏移
        color = 'blue' if is_in else 'red'
        if color == 'blue':
            ax.plot([x0, x1p], [y0, y1p], color=color, linewidth=1)
        ax.scatter(x0, y0, color=color, s=10)
        ax.scatter(x1p, y1p, color=color, s=10)

    """
    for (x0, y0), (x1, y1), is_in in zip(match_kp0, match_kp1, in_target):
        x1 += image0.shape[1]
        color = 'blue' if is_in else 'red'
        if color == 'blue':
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=1)
        ax.scatter(x0, y0, color=color, s=10)
        ax.scatter(x1, y1, color=color, s=10)
    """
    plt.savefig("demo_output.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    path0 = "./Datasets/SPUS/DIFF/S1602/imgs/00005.jpg"
    mask0 = "./Datasets/SPUS/DIFF/S1602/mask/00005_mask.png"
    path1 = "./Datasets/SPUS/DIFF/S1602/imgs/00005.jpg"
    mask1 = "./Datasets/SPUS/DIFF/S1602/mask/00005_mask.png"

    match(path0, path1, mask1, target_class="VA")
