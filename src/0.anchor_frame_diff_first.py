"""
在候选帧中尝试找出能覆盖最大差异的子集，而不是顺序判断加入。
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
    改进逻辑：优先选择与已选关键帧差异最大的帧，提升多样性。
    """
    image_paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    if not image_paths:
        return []

    # === Step 1: 计算熵值 ===
    frame_info_list = []
    for idx, path in enumerate(image_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        entropy = compute_entropy(img)
        frame_info_list.append((idx, path, entropy, img))

    if not frame_info_list:
        return []

    # === Step 2: 根据熵选候选帧 ===
    frame_info_list.sort(key=lambda x: x[2], reverse=True)
    if use_percentile:
        all_entropies = [x[2] for x in frame_info_list]
        entropy_threshold = np.percentile(all_entropies, (1 - top_percent) * 100)
        candidate_frames = [x for x in frame_info_list if x[2] >= entropy_threshold]
    else:
        top_n = int(np.ceil(len(frame_info_list) * top_percent))
        candidate_frames = frame_info_list[:top_n]

    # === Step 3: 贪心选择差异最大的一组帧 ===
    key_frames = []
    used = [False] * len(candidate_frames)

    # 先选熵值最高的第一个
    key_frames.append(candidate_frames[0])
    used[0] = True

    while len(key_frames) < max_keyframes and not all(used):
        max_diff = -1
        next_idx = -1
        for i, (idx, path, entropy, img) in enumerate(candidate_frames):
            if used[i]:
                continue
            # 与所有已选关键帧的最小相似度
            min_ssim = min(ssim(img, kf[3]) for kf in key_frames)
            diff = 1 - min_ssim
            if diff > max_diff:
                max_diff = diff
                next_idx = i
        if max_diff >= ssim_diff_threshold and next_idx != -1:
            key_frames.append(candidate_frames[next_idx])
            used[next_idx] = True
        else:
            break  # 没有更多足够不同的帧了

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