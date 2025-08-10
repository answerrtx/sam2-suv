import os
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import onnxruntime as ort
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="LightGlue keypoint matching")
    parser.add_argument("--image_folder", required=True, help="Folder containing query images")
    parser.add_argument("--ref_img_path", required=True, help="Support image path")
    parser.add_argument("--ref_mask_path", required=True, help="Support mask path")
    parser.add_argument("--color_map_path", required=True, help="Path to color map json")
    parser.add_argument("--click_save_folder", required=True, help="Output folder for click results")
    parser.add_argument("--anchor_file", required=True, help="List of image filenames to process")
    parser.add_argument("--target_class", required=True, help="Target class name for classification")
    parser.add_argument("--lg_export", required=False, default="./checkpoints/superpoint_lightglue_pipeline.ort.onnx")
    return parser.parse_args()


def preprocess_image(image_path, size=1024):
    img = Image.open(image_path).convert("L")
    orig_size = img.size  # (W, H)
    img = img.resize((size, size), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    return img_np[None, None, :, :], np.array(img), orig_size


def run_lightglue_onnx(img0_path, img1_path, model_path):
    img0, img0_resized, img0_size = preprocess_image(img0_path)
    img1, img1_resized, img1_size = preprocess_image(img1_path)
    input_tensor = np.concatenate([img0, img1], axis=0)
    sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    keypoints, matches, scores = sess.run(output_names, {input_name: input_tensor})

    kp0 = keypoints[0]
    kp1 = keypoints[1]
    match_coords = []
    for (batch_idx, idx0, idx1), score in zip(matches, scores):
        if batch_idx == 0:
            pt0 = kp0[idx0]  # query
            pt1 = kp1[idx1]  # support
            match_coords.append((pt0, pt1, score))
    return match_coords, img0_resized, img0_size, img1_size


def main():
    args = parse_args()
    os.makedirs(args.click_save_folder, exist_ok=True)
    class_name = args.target_class
    with open(args.color_map_path, "r") as f:
        color_map = json.load(f)
        color_map = {tuple(map(int, k.strip("()").split(","))): v for k, v in color_map.items()}
        mc_color = [k for k, v in color_map.items() if v == "Deltoid"][0]#args.target_class][0]

    mask_img = Image.open(args.ref_mask_path).convert("RGB")
    mask_np = np.array(mask_img)
    mc_mask = np.all(mask_np == mc_color, axis=-1).astype(np.uint8)

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

        matches, img0_resized, img0_size, img1_size = run_lightglue_onnx(image_path, args.ref_img_path, args.lg_export)
        labels = []
        H0, W0 = img0_resized.shape
        W1, H1 = img1_size

        for pt0, pt1, score in matches:
            if score < 0.02:
                continue
            x1, y1 = pt1
            x1_orig = int(round(x1 / 1024 * W1))
            y1_orig = int(round(y1 / 1024 * H1))

            if 0 <= y1_orig < mc_mask.shape[0] and 0 <= x1_orig < mc_mask.shape[1]:
                label = 1 if mc_mask[y1_orig, x1_orig] > 0 else 0
            else:
                label = 0

            x0, y0 = pt0
            x0_orig = x0 / 1024 * img0_size[0]
            y0_orig = y0 / 1024 * img0_size[1]
            labels.append((x0_orig, y0_orig, label))

        if len(labels) < 2:
            continue

        save_txt = os.path.join(args.click_save_folder, f"{stem}.txt")
        with open(save_txt, "w") as f:
            for x, y, label in labels:
                f.write(f"{x:.2f},{y:.2f},{label}\n")

        vis_img = cv2.imread(image_path)
        for x, y, label in labels:
            x, y = int(x), int(y)
            color = (0, 0, 255) if label == 0 else (255, 255, 0)
            cv2.circle(vis_img, (x, y), 2, color, -1)
        plt.imsave(os.path.join(args.click_save_folder, f"{stem}_vis.png"), vis_img[..., ::-1])

    print("✅ LightGlue 匹配与标注完成，结果已保存。")


if __name__ == "__main__":
    main()
