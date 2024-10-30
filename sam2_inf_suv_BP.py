#conda install python==3.12 pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


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


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def read_positive_prompt(filename):
    coordinates = []
    with open(filename, 'r') as file:
        cnt = 0
        for line in file:
            if cnt == 0:
                cnt+=1
                continue
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


def run():
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    from sam2.build_sam import build_sam2_video_predictor

    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


    # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
    video_dir = "./Datasets/CLU/MS011"
    result_dir = video_dir + '_RESULT'
    boxes = read_boxes('./Datasets/CLU/MS011_ROIS_AF/00012.txt')
    positive_prompts = read_positive_prompt('./Datasets/CLU/MS011_MATCHES/00012_00013.txt')
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # take a look the first video frame
    frame_idx = 0
    #plt.figure(figsize=(9, 6))
    #plt.title(f"frame {frame_idx}")
    #plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))


    inference_state = predictor.init_state(video_path=video_dir)    
    predictor.reset_state(inference_state)

    ann_frame_idx = 5  # the frame index we interact with

    # Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
    # sending all clicks (and their labels) to `add_new_points_or_box`

    # Let's add a positive click at (x, y) = (460, 60) to refine the mask

    # note that we also need to send the original box input along with
    # the new refinement click together into `add_new_points_or_box`

    ann_obj_id = 12  # give a unique id to each object we interact with (it can be any integers)

    # for labels, `1` means positive click and `0` means negative click
    new_n=[1]*len(positive_prompts)
    #new_n.append(0)
    #new_n.append(0)
    new_n.append(0)
    new_n.append(0)
    new_n.append(0)
    new_n.append(0)
    P_labels = np.array(new_n, np.int32)

    #positive_prompts.append([223,174.5])
    #positive_prompts.append([471.6,188.3])
    positive_prompts.append([242,62.6])
    positive_prompts.append([328,131])
    positive_prompts.append([276,121])
    positive_prompts.append([306,118])
    P_points = np.array(positive_prompts, dtype=np.float32)
    print(len(P_points),len(new_n))



    for i in range(len(boxes)):
        box = np.array(boxes[i], dtype=np.float32)
        
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=P_points,
            labels=P_labels,
            box=box,
        )

        # show the results on the current (interacted) frame
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_box(box, plt.gca())
        show_points(P_points, P_labels, plt.gca())
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        plt.show()
        
    negative_points = np.array([[223,174.5], [471.6,188.3]])
    negative_labels = np.array([0]*len(negative_points), np.int32)

'''    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=negative_points,
        labels=negative_labels,
        #box=box,
    )

    # show the results on the current (interacted) frame
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    #show_box(box, plt.gca())
    show_points(negative_points, negative_labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    plt.show()    '''

'''    points = np.array([[190, 190], [150, 150], [250, 250], [250,150], [325,200], [350,190], [300, 150], [250,200]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1, 0, 0, 1, 1, 0, 0, 0], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # show the results on the current (interacted) frame
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    plt.show()'''

'''    # for next frames
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    vis_frame_stride = 30
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

    plt.show()'''


run()