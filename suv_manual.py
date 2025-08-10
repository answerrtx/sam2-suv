import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import sys
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
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def read_positive_prompt(filename):
    coordinates = []
    with open(filename, 'r') as file:
        for line in file:
            coords = list(map(float, line.strip().split(',')))
            coordinates.append(coords)
    return coordinates

def read_negative_prompt(filename):
    coordinates = []
    with open(filename, 'r') as file:
        for line in file:
            coords = list(map(float, line.strip().split(',')))
            coordinates.append(coords)
    return coordinates

def read_positive_prompt(filename):
    coordinates = []
    with open(filename, 'r') as file:
        for line in file:
            coords = list(map(float, line.strip().split(',')))
            coordinates.append(coords)
    return coordinates


def read_boxes(filename):
    coordinates = []
    with open(filename, 'r') as file:
        for line in file:
            coords = list(map(float, line.strip().split()))
            coordinates.append(coords)
    return coordinates

def points2box(points_list,label_list):
    pos_points = [pt for pt, label in zip(points_list, label_list) if label == 1]    
    if not points_list:
        return None  # 或者返回 (0, 0, 0, 0)

    x_coords = [pt[0] for pt in pos_points]
    y_coords = [pt[1] for pt in pos_points]

    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)

    return (xmin, ymin, xmax, ymax)

def get_candicate_points(anno_frame_idx):
    if anno_frame_idx == 0:
        return [[210, 350], [191, 227],[386,254],[435,229],[212,350],
                [304,271],[431,342],[302,191],[289,128],
                [397,192]]
    else: 
        return [[210, 350], [250, 220],[260,162]]

def classify_clicks(ann_frame_idx, ann_frame_candicate):
    points = []
    labels = []
    file_path = str(ann_frame_idx)+"_click_points.txt"  # 你保存点击点的路径

    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在，返回空列表")
        return [], []

    with open(file_path, "r") as f:
        for line in f:
            try:
                x_str, y_str, label_str = line.strip().split(",")
                x = float(x_str)
                y = float(y_str)
                label = int(label_str)
                points.append([int(round(x)), int(round(y))])
                labels.append(label)
            except ValueError:
                print(f"跳过无效行: {line.strip()}")
                continue

    return points, labels


# 鼠标点击回调函数
def onclick(event):
    global click_coords
    if event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata

        if event.button == 1:  # 左键
            label = 1
            plt.gca().scatter(x, y, color='green', marker='*', s=100, edgecolor='white')
            print(f"左键点击: ({x:.2f}, {y:.2f}) -> 正样本")
        elif event.button == 3:  # 右键
            label = 0
            plt.gca().scatter(x, y, color='red', marker='*', s=100, edgecolor='white')
            print(f"右键点击: ({x:.2f}, {y:.2f}) -> 负样本")
        else:
            return  # 忽略中键等

        click_coords.append((x, y, label))
        plt.draw()

if __name__ == "__main__":
    op_type = int(sys.argv[1])
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        # Use bfloat16 for CUDA devices
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    #op_type = 1
    # Initialize SAM2
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
    video_dir = "./Datasets/dev_test/hnd3"
    video_result = "./Output/dev_test/hnd3"
    os.makedirs(video_result, exist_ok=True)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


    #plt.title(f"frame {frame_idx}")
    #plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
    #plt.show()
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)


    # select anchor frame
    anchor_frame_list = [10,23,33]

    # obtain query text
    query_text = ""
    # extract query text feature
    # generate candidate clicks for each anchor frame
    for anno_frame in anchor_frame_list:
        
        #anno_frame_candicate = get_candicate_points(anno_frame)
        #positive, negative = classify_clicks(anno_frame_candicate)
        ann_frame_idx = anno_frame # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        # Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
        # sending all clicks (and their labels) to `add_new_points_or_box`
        point_list = get_candicate_points(ann_frame_idx)
        point_list, labels_list = classify_clicks(ann_frame_idx, point_list)
        print(point_list,labels_list)
        # for labels, `1` means positive click and `0` means negative click
        box_list = points2box(point_list, labels_list)
        box = np.array(box_list, dtype=np.float32)
        points = np.array(point_list, dtype=np.float32)
        labels = np.array(labels_list, np.int32)
        #(x_min, y_min, x_max, y_max)
        if op_type == 1:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box,
            )
            plt.figure(figsize=(9, 6))
            
            plt.title(f"frame {ann_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
            show_points(points, labels, plt.gca())
            show_box(box, plt.gca())
            show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
            plt.show()

        else:
            click_coords = []
            # show the results on the current (interacted) frame
            fig, ax = plt.subplots(figsize=(9, 6))
            plt.title(f"frame {ann_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))

            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
            with open(str(anno_frame)+"_click_points.txt", "w") as f:
                for x, y, label in click_coords:
                    f.write(f"{x:.2f}, {y:.2f}, {label}\n")
            print("点击点已保存至 click_points.txt")

    aa = 1
    if aa!=0 and op_type == 1: 
       
        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,reverse=False):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        # render the segmentation results every few frames
        vis_frame_stride = 2
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
        # extract 
