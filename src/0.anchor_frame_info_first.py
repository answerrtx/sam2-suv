"""
从熵高的帧中筛出候选池。

然后逐个遍历，只要与已有关键帧的 SSIM 差异 > 阈值，就加入。
"""

import os
import glob
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_entropy(gray_img):
    """
    计算图像的熵，作为信息量的指标。
    """
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist_norm = hist / (gray_img.shape[0] * gray_img.shape[1])
    hist_norm += 1e-12  # 避免log(0)
    entropy = -np.sum(hist_norm * np.log2(hist_norm))
    return entropy

def pick_frames(folder, top_percent=0.1, ssim_diff_threshold=0.02, max_keyframes=None, use_percentile=True):
    """
    改进后的多关键帧筛选方法：
    1. 根据熵值选择候选帧（前top_percent或分位数阈值以上）
    2. 迭代筛选出与所有已选关键帧差异足够大的帧
    """
    image_paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    #print(image_paths)

    if not image_paths:
        return []
    # 读取所有帧并计算熵值
    frame_info_list = []
    for idx, path in enumerate(image_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        info_val = compute_entropy(img)
        frame_info_list.append((idx, path, info_val, img))

    if not frame_info_list:
        return []
    frame_info_list.sort(key=lambda x: x[2], reverse=True)  # 按熵降序排序
    # 确定候选池
    if use_percentile:
        all_entropies = [x[2] for x in frame_info_list]
        percentile = (1 - top_percent) * 100  # 转换为分位数
        entropy_threshold = np.percentile(all_entropies, percentile)
        candidate_frames = [x for x in frame_info_list if x[2] >= entropy_threshold]
    else:
        top_n = int(np.ceil(len(frame_info_list) * top_percent))
        candidate_frames = frame_info_list[:top_n]
    #print(len(candidate_frames))

    key_frames = []
    for candidate in candidate_frames:
        if len(key_frames) >= max_keyframes:
            break
        idx, path, info_val, gray_img = candidate
        # 检查与所有已选关键帧的差异
        qualified = True
        for kf in key_frames:
            ssim_score, _ = ssim(gray_img, kf[3], full=True)
            if (1 - ssim_score) <= ssim_diff_threshold:
                qualified = False
                break
            else:
                pass
                #print("===",ssim_score)
        if qualified:
            key_frames.append(candidate)

    # 按帧索引排序输出
    key_frames.sort(key=lambda x: x[0])
    return [(k[0], k[1], k[2]) for k in key_frames]

import argparse

# 示例使用
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Select anchor frames from video folder")
    parser.add_argument("--folder", type=str, required=True, help="Path to the image folder")
    parser.add_argument("--top_percent", type=float, default=0.25, help="Top entropy percentage to consider")
    parser.add_argument("--ssim_diff_threshold", type=float, default=0.2, help="SSIM difference threshold")
    parser.add_argument("--max_keyframes", type=int, default=4, help="Maximum number of keyframes")
    parser.add_argument("--use_percentile", type=bool, default=False, help="Use percentile-based entropy threshold")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the keyframes")
    args = parser.parse_args()
    folder = args.folder
    folder_name = folder.split("/")[-1]
    img_folder = os.path.join(folder, "imgs")
    print(img_folder,"image_folder!")
    key_frames = pick_frames(
        img_folder,
        top_percent=args.top_percent,
        ssim_diff_threshold=args.ssim_diff_threshold,
        max_keyframes=args.max_keyframes,
        use_percentile=args.use_percentile
    )
    print("选出的关键帧：")
    for idx, path, info in key_frames:
        print(f"帧 {idx}: 路径= {path}, 熵={info:.2f}")

    import shutil
    # 保存关键帧索引到文本文件
    output_path = args.output_path
    output_folder = output_path.replace(".txt", "/")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    with open(output_path, "w") as f:
        for idx, path, _ in key_frames:
            filename = os.path.basename(path)
            f.write(f"{filename}\n")
            src_path = os.path.join(img_folder, filename)
            dst_path = os.path.join(output_folder, filename)  # 保留原文件名
            shutil.copy(src_path, dst_path)

    print(f"关键帧索引已保存到: {output_path}")