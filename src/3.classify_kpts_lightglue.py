import os
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import onnxruntime as ort
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



def parse_args():
    parser = argparse.ArgumentParser(description="LightGlue keypoint matching")
    parser.add_argument("--image_folder", required=True, help="Folder containing query images")
    parser.add_argument("--ref_img_path", required=True, help="Support image path")
    parser.add_argument("--ref_mask_path", required=True, help="Support mask path")
    parser.add_argument("--color_map_path", required=True, help="Path to color map json")
    parser.add_argument("--click_save_folder", required=True, help="Output folder for click results")
    parser.add_argument("--anchor_file", required=True, help="List of image filenames to process")
    parser.add_argument("--target_class", required=True, help="Target class name for classification")
    parser.add_argument("--lg_export", required=False, default=None)
    parser.add_argument("--sp_export", required=False, default=None)
    parser.add_argument("--kpt_folder", required=False, default=None, help="Folder to load keypoints")
    parser.add_argument("--subregion", type=str, default=None, required=True)

    return parser.parse_args()


def preprocess_image(image_path, size=1024):
    img = Image.open(image_path).convert("L")
    orig_size = img.size  # (W, H)
    img = img.resize((size, size), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    return img_np[None, None, :, :], np.array(img), orig_size


def run_lightglue_onnx(img0_path, img1_path, lg_path, sp_path, match_mask=None, ref_mask=None):
    extractor_type = "superpoint"  
    extractor_path = sp_path
    lightglue_path = lg_path

    img0, img0_resized, img0_size = preprocess_image(img0_path)
    img1, img1_resized, img1_size = preprocess_image(img1_path)
    print(img0_path, img1_path, "============")
    
    from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
    from lightglue.utils import load_image, rbd
    from lightglue import viz2d

    # SuperPoint+LightGlue
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)  # load the extractor
    matcher = LightGlue(features='superpoint').eval().to(device)  # load the matcher


    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
    image0 = load_image(img0_path)#.cuda()
    image1 = load_image(img1_path)#.cuda()

    # extract local features
    #feats0 = extractor.extract(image0.to(device))  # auto-resize the image, disable with resize=None
    #feats1 = extractor.extract(image1.to(device))

    feats0 = extractor.extract(image0.to(device), mask=match_mask.to(device))
    feats1 = extractor.extract(image1.to(device), mask=ref_mask.to(device))

    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([image0, image1])
    viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
    plt.show()

from collections import defaultdict
import cv2
def main():
    args = parse_args()
    os.makedirs(args.click_save_folder, exist_ok=True)
    class_name = args.target_class
    with open(args.color_map_path, "r") as f:
        color_map = json.load(f)
        color_map = {tuple(map(int, k.strip("()").split(","))): v for k, v in color_map.items()}
        mc_color = [k for k, v in color_map.items() if v == args.target_class][0]

    mask_img = Image.open(args.ref_mask_path).convert("RGB")
    mask_np = np.array(mask_img)
    mc_mask = np.all(mask_np == mc_color, axis=-1).astype(np.uint8)
    image_folder = os.path.join(args.image_folder, "imgs")
    ref_mask1 = mc_mask.astype(np.uint8) * 255

    support_img = Image.open(args.ref_img_path)
    support_w, support_h = support_img.size

    image_folder = os.path.join(args.image_folder, "imgs")

    with open(args.anchor_file, "r") as f:
        frame_list = [line.strip() for line in f if line.strip().endswith(".jpg")]

    for fname in frame_list:
        stem = os.path.splitext(fname)[0]
        image_path = os.path.join(image_folder, fname)
        if not os.path.exists(image_path):
            print(f"❌ 缺失帧：{fname}")
            continue
    
        # 加载 image0 的尺寸信息
        image0_pil = Image.open(image_path).convert("RGB")
        h, w = image0_pil.height, image0_pil.width
        image0 = np.array(image0_pil)

        # === 加载 cluster keypoints 文件（和 OmniGlue 相同逻辑） ===
        kpt_path = os.path.join(args.kpt_folder, f"{stem}.txt")
        if args.subregion!=None and args.kpt_folder:

            if not os.path.exists(kpt_path):
                print(f"⚠️ 缺失 cluster 信息文件：{kpt_path}")
                continue

            with open(kpt_path, "r") as f:
                cluster_points = defaultdict(list)
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 3:
                        x, y, cluster = float(parts[0]), float(parts[1]), parts[2].strip()
                        cluster_points[cluster].append([x, y])

            best_cluster = None
            best_score = -1
            best_mask = None

            for cluster_name, points in cluster_points.items():
                if len(points) < 3:
                    continue

                # === 计算 cluster mask ===
                cluster_mask = np.zeros((h, w), dtype=np.uint8)
                pts = np.array(points, dtype=np.int32)
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(cluster_mask, hull, 255)
            
                # === 转为 tensor mask 传给 SuperPoint ===
                match_mask_tensor = torch.from_numpy(cluster_mask / 255.0).float()
                ref_mask_tensor = torch.from_numpy(ref_mask1 / 255.0).float()

                # === 调用 LightGlue 特征提取与匹配 ===
                run_lightglue_onnx(
                    image_path,
                    args.ref_img_path,
                    lg_path=args.lg_export,
                    sp_path=args.sp_export,
                    match_mask=match_mask_tensor,
                    ref_mask=ref_mask_tensor,
                )

    print("✅ LightGlue 匹配与标注完成，结果已保存。")


if __name__ == "__main__":
    main()
