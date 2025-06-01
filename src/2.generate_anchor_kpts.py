import os
import argparse
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import sys
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omniglue_onnx import omniglue
from omniglue_onnx.omniglue import utils

import onnxruntime as ortt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import torch
import torchvision.transforms as T
from sklearn.cluster import MeanShift, estimate_bandwidth

# 添加 transform 用于 DINOv2 特征提取
transform_dino = T.Compose([
    T.Resize((518, 518)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
# 显式使用 GPU（CUDA）

# ---------- 函数定义 ----------

def compute_ssim(img1, img2):
    img1_gray = np.array(Image.fromarray(img1).convert("L"))
    img2_gray = np.array(Image.fromarray(img2).convert("L"))
    return ssim(img1_gray, img2_gray)

import matplotlib.pyplot as plt
import numpy as np

def visualize_keypoints_by_cluster(image, keypoints, labels, radius=4, cmap_name="tab20"):
    """
    可视化聚类关键点，每个 cluster 用不同颜色。

    Args:
        image: 原始图像 (H,W,3)
        keypoints: [(x,y), ...] 坐标数组，长度为 N
        labels: 每个点对应的聚类标签（长度 N）
        radius: 每个点绘制半径
        cmap_name: matplotlib colormap 名称

    Returns:
        彩色标注的图像（np.array）
    """
    img = image.copy()
    cmap = plt.get_cmap(cmap_name)
    unique_labels = sorted(set(labels))
    color_map = {
        label: tuple((np.array(cmap(i % cmap.N)[:3]) * 255).astype(np.uint8))
        for i, label in enumerate(unique_labels)
    }

    for (x, y), label in zip(keypoints, labels):
        color = color_map[label]
        x, y = int(round(x)), int(round(y))
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    if 0 <= x + dx < img.shape[1] and 0 <= y + dy < img.shape[0]:
                        img[y + dy, x + dx] = color

    return img

import matplotlib.pyplot as plt
import numpy as np

def interactive_pca_matplotlib(reduced, keypoints, image):
    """
    使用 matplotlib 实现 PCA 投影图交互，
    点击 PCA 点后在原图中高亮对应关键点。
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c='blue', s=30, picker=True)
    ax.set_title("PCA Projection (Click to View Keypoint)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # 将点击与 index 对应起来
    def on_pick(event):
        ind = event.ind[0]  # 多点时只取第一个
        print(f"🟢 点击了点 {ind}")

        # 在原图中标出 keypoint
        img_copy = image.copy()
        x, y = int(round(keypoints[ind][0])), int(round(keypoints[ind][1]))
        img_copy[y-4:y+5, x-4:x+5] = [255, 0, 0]  # 红色标记

        # 显示图像
        plt.figure(figsize=(5, 5))
        plt.imshow(img_copy)
        plt.title(f"Keypoint {ind} on Original Image")
        plt.axis('off')
        plt.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()



from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
def main():
    parser = argparse.ArgumentParser(description="Initial Anchor Descriptors with OmniGlue")
    parser.add_argument("--image_folder", required=True, help="Path to image folder")
    parser.add_argument("--index_file", required=True, help="Path to keyframe index file")
    parser.add_argument("--output", required=True, help="Output folder for match results")

    # 模型路径
    parser.add_argument("--og_export", required=True, help="Path to OmniGlue ONNX export")
    parser.add_argument("--sp_export", required=True, help="Path to SuperPoint ONNX export")
    parser.add_argument("--dino_export", required=True, help="Path to DINOv2 weights")

    # SSIM 匹配阈值
    parser.add_argument("--ssim_min", type=float, default=0.8, help="Minimum SSIM for matching")
    parser.add_argument("--ssim_max", type=float, default=0.95, help="Maximum SSIM for matching")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    image_folder = os.path.join(args.image_folder, "imgs")
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
    image_paths = [os.path.join(image_folder, f) for f in image_files]
    file_to_index = {fname: idx for idx, fname in enumerate(image_files)}

    # 初始化模型
    og = omniglue.OmniGlue(
        og_export=args.og_export,
        sp_export=args.sp_export,
        dino_export=args.dino_export,
        providers= ['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    with open(args.index_file, "r") as f:
        frame_filenames = [line.strip() for line in f if line.strip().endswith(".jpg")]

    for fname in frame_filenames:
        if fname not in file_to_index:
            print(f"⚠️ 文件 {fname} 不在图像列表中，跳过")
            continue

        index = file_to_index[fname]
        base_img_path = image_paths[index]
        base_img = np.array(Image.open(base_img_path).convert("RGB"))

        best_sim = -1
        best_index = None

        for offset in range(1, 10):
            for direction in [-1, 1]:
                candidate_idx = index + direction * offset
                if 0 <= candidate_idx < len(image_paths):
                    candidate_img = np.array(Image.open(image_paths[candidate_idx]).convert("RGB"))
                    score = compute_ssim(base_img, candidate_img)
                    if args.ssim_min <= score <= args.ssim_max and score > best_sim:
                        best_sim = score
                        best_index = candidate_idx

        if best_index is None:
            print(f"[{fname}] ❌ 无匹配帧")
            continue

        image0 = np.array(Image.open(base_img_path).convert("RGB"))
        image1 = np.array(Image.open(image_paths[best_index]).convert("RGB"))
        print(base_img_path," to_match: ",best_index)
        match_kp0, match_kp1, confidences = og.FindMatches(image0, image1)
        keep = confidences > 0.02
        match_kp0 = match_kp0[keep]
        match_kp1 = match_kp1[keep]

        # Step 1: 提取图像特征
        feat_map = og.dino_extract.forward(image0)  # image0 是 numpy array (H,W,3)
        h_feat, w_feat, c_feat = feat_map.shape
        # Step 2: 坐标映射到 patch index
        def get_patch_features(feat_map, keypoints, image_shape):
            H, W = image_shape[:2]
            feat_H, feat_W = feat_map.shape[:2]
            descriptors = []
            for x, y in keypoints:
                px = int(x / W * feat_W)
                py = int(y / H * feat_H)
                px = np.clip(px, 0, feat_W - 1)
                py = np.clip(py, 0, feat_H - 1)
                descriptors.append(feat_map[py, px, :])
            return np.array(descriptors)

        selected_features = get_patch_features(feat_map, match_kp0, image0.shape)


        # Step 3: PCA 降维
        pca = PCA(n_components=8)
        reduced_pca = pca.fit_transform(selected_features)
        tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
        reduced = tsne.fit_transform(selected_features)

        #plot_pca_with_image_mapping(reduced, match_kp0, image0)
        #interactive_pca_keypoint_view(reduced, match_kp0, image0)
        #interactive_pca_matplotlib(reduced, match_kp0, image0)
        print(selected_features.shape, reduced.shape,match_kp0.shape)
        # Step 4: 聚类
        joint_feat = np.concatenate([
            selected_features,
            np.array(match_kp0) * 0.05  # 位置缩放以保持尺度匹配
        ], axis=1)
        from sklearn.cluster import KMeans
        cluster = KMeans(n_clusters=5).fit(joint_feat)
        labels = cluster.labels_

        # Step 5: 保存带聚类标签的结果
        save_path = os.path.join(args.output, f"{os.path.splitext(fname)[0]}.txt")
        with open(save_path, "w") as f:
            for (x, y), lbl in zip(match_kp0, labels):
                label_str = f"cluster{lbl}" if lbl != -1 else "noise"
                f.write(f"{x:.2f},{y:.2f},{label_str}\n")
        viz = utils.visualize_matches(image0, image1, match_kp0, match_kp1, 
                              np.eye(match_kp0.shape[0]), title=f"{len(match_kp0)} matches")
        img0_keypoints = visualize_keypoints_by_cluster(image0, match_kp0, labels)

        concat_img = np.concatenate([img0_keypoints, viz], axis=1)
        plt.imsave(os.path.join(args.output, f"{os.path.splitext(fname)[0]}_cluster.png"), img0_keypoints)

    print("🎯 匹配完成 ✅")

if __name__ == "__main__":
    main()
