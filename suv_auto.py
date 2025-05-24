import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from PIL import Image as PILImage

import argparse
import json

# 加载颜色映射 (从字符串 RGB 到 label)
with open('./suv.json', 'r') as f:
    suv_data = json.load(f)

# 将 "(R, G, B)" 形式字符串转换为 RGB 元组 -> label 映射
color_map = {eval(k): v for k, v in suv_data.get('color_map', {}).items()}
label_map = {v: eval(k) for k, v in suv_data.get('color_map', {}).items()}
def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM2 SUV segmentation.")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--video_result", required=True)
    parser.add_argument("--click_folder", required=True)
    parser.add_argument("--seg_model", required=True)  # 暂时仅支持 sam2
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_cfg", required=True)
    parser.add_argument("--prompt_type", required=True)  
    parser.add_argument("--target_class", required=True)
    parser.add_argument("--augmentation", default="False", help="Use augmentation for points")
    return parser.parse_args()

def save_binary_mask(mask_np, save_path):
    # 强制 squeeze 到 (H, W)，确保是二维图像
    mask_np = np.squeeze(mask_np)

    if mask_np.dtype != np.uint8:
        mask_np = (mask_np > 0).astype(np.uint8) * 255  # 变为 0/255 图像

    if mask_np.ndim != 2:
        raise ValueError(f"Mask shape must be 2D after squeeze, got {mask_np.shape}")

    pil_img = Image.fromarray(mask_np)
    pil_img.save(save_path)


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size,
               edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size,
               edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                 facecolor=(0, 0, 0, 0), lw=2))


def points2box(points_list, label_list):
    pos_points = [pt for pt, label in zip(points_list, label_list) if label == 1]
    if not pos_points:
        return None
    x_coords = [pt[0] for pt in pos_points]
    y_coords = [pt[1] for pt in pos_points]
    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)
    return (xmin, ymin, xmax, ymax)


def get_click_data(frame_idx, click_folder):
    txt_path = os.path.join(click_folder, f"{frame_idx:05d}.txt")
    point_list = []
    label_list = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                x, y, label = map(float, parts)
                point_list.append([x, y])
                label_list.append(int(label))
    return point_list, label_list


if __name__ == "__main__":
    args = parse_args()
    video_dir = os.path.join(args.video_dir,'imgs')
    video_result = args.video_result
    click_folder = args.click_folder
    sam2_checkpoint = args.checkpoint
    model_cfg = args.model_cfg
    prompt_type = args.prompt_type
    seg_model = args.seg_model
    augmentation = args.augmentation
    if seg_model != "sam2":
        raise NotImplementedError("Only SAM2 is supported in this script.")


    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print("\nMPS support is experimental and may produce different results.")

    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


    os.makedirs(video_result, exist_ok=True)

    frame_names = sorted([
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ], key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    click_files = sorted([f for f in os.listdir(click_folder) if f.endswith(".txt")])
    anchor_frame_list = [int(os.path.splitext(f)[0].lstrip("0") or "0") for f in click_files]

    for anno_frame in anchor_frame_list:
        ann_frame_idx = anno_frame
        ann_obj_id = 1

        point_list, labels_list = get_click_data(ann_frame_idx, click_folder)
        print(f"Frame {ann_frame_idx}: {len(point_list)} points, {sum(labels_list)} positive")

        box_list = points2box(point_list, labels_list)
        if box_list is None:
            continue
        box = np.array(box_list, dtype=np.float32)
        points = np.array(point_list, dtype=np.float32)
        labels = np.array(labels_list, np.int32)

        if np.sum(labels == 1) < 1: continue

        
        #get_surrounding np
        # 加载图像大小
        img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
        img_h, img_w = Image.open(img_path).size[::-1]  # H, W
        if args.augmentation == "True":
            print("Using augmentation for points")
            # ==== 1. 正点扩增：每个正点周围采样两个新正点 ====
            aug_pos_points = []
            for pt, label in zip(point_list, labels_list):
                if label != 1:
                    continue
                for _ in range(2):  # 每个正点采样两个
                    offset = np.random.randint(-10, 10, size=2)
                    new_pt = [pt[0] + offset[0], pt[1] + offset[1]]
                    if 0 <= new_pt[0] < img_w and 0 <= new_pt[1] < img_h:
                        aug_pos_points.append(new_pt)
            point_list.extend(aug_pos_points)
            labels_list.extend([1] * len(aug_pos_points))

            # ==== 2. 负点采样：在bbox外围5-10像素生成5个 ====
            xmin, ymin, xmax, ymax = map(int, box_list)
            neg_points = []
            max_attempt = 100
            neg_count = 0
            while neg_count < 5 and max_attempt > 0:
                max_attempt -= 1
                pad = np.random.randint(5, 11)
                side = np.random.choice(['top', 'bottom', 'left', 'right'])
                if side == 'top':
                    x = np.random.randint(xmin - pad, xmax + pad + 1)
                    y = ymin - pad
                elif side == 'bottom':
                    x = np.random.randint(xmin - pad, xmax + pad + 1)
                    y = ymax + pad
                elif side == 'left':
                    x = xmin - pad
                    y = np.random.randint(ymin - pad, ymax + pad + 1)
                else:  # right
                    x = xmax + pad
                    y = np.random.randint(ymin - pad, ymax + pad + 1)
                if 0 <= x < img_w and 0 <= y < img_h:
                    neg_points.append([x, y])
                    neg_count += 1
            point_list.extend(neg_points)
            labels_list.extend([0] * len(neg_points))

            #print(point_list, labels_list)
            points2 = np.array(point_list, dtype=np.float32)
            labels2 = np.array(labels_list, np.int32)
        # annotate points
        else:
            points2 = points
            labels2 = labels

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points2,
            labels=labels2,
        )
        if args.prompt_type == "point_box":
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box,
            )

        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_points(points2, labels2, plt.gca())
        if args.prompt_type == "point_box":
            show_box(box, plt.gca())
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        #plt.show()

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=False):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        
    vis_frame_stride = 1
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        save_path = os.path.join(video_result, f"frame_{out_frame_idx}.png")
        plt.savefig(save_path)
        plt.close()
        save_path = os.path.join(video_result, f"frame_{out_frame_idx}.png")
    plt.savefig(save_path)
    plt.close()
    

    """for frame_idx, mask_dict in video_segments.items():
        frame_base = os.path.splitext(frame_names[frame_idx])[0]  # 如 00001
        #frame_out_dir = os.path.join(video_result, frame_base)
        #os.makedirs(frame_out_dir, exist_ok=True)

        for obj_id, mask_np in mask_dict.items():
            out_path = os.path.join(video_result, f"{frame_idx}.png")
            save_binary_mask(mask_np, out_path)"""