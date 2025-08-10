import onnxruntime as ort
import numpy as np
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt

def preprocess_image(image_path, size=(1024, 1024)):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize(size)
    img_np = np.array(img).astype(np.float32) / 255.0  # Normalize
    img_np = img_np[None, None, :, :]  # Shape: (1, 1, H, W)
    return img_np, np.array(img)  # also return for visualization

def run_lightglue_onnx(img0_path, img1_path, model_path="./superpoint_lightglue_pipeline.ort.onnx"):
    # Preprocess and load images
    img0_tensor, img0_vis = preprocess_image(img0_path)
    img1_tensor, img1_vis = preprocess_image(img1_path)

    # Interleave input
    input_tensor = np.concatenate([img0_tensor, img1_tensor], axis=0)  # shape: (2, 1, H, W)

    # Inference
    sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    outputs = sess.run(output_names, {input_name: input_tensor})
    keypoints, matches, scores = outputs

    # Extract match pairs
    kp0 = keypoints[0]  # keypoints from img0
    kp1 = keypoints[1]  # keypoints from img1
    match_coords = []
    for (batch_idx, idx0, idx1), score in zip(matches, scores):
        if batch_idx == 0:
            pt0 = kp0[idx0]
            pt1 = kp1[idx1]
            match_coords.append((pt0, pt1, score))

    return match_coords, img0_vis, img1_vis

def draw_matches(img0, img1, matches, save_path="lightglue_matches.png"):
    # Stack images side by side
    h, w = img0.shape
    canvas = np.concatenate([img0, img1], axis=1)  # (H, 2W)
    canvas_color = np.stack([canvas]*3, axis=-1)  # convert to RGB

    plt.figure(figsize=(14, 7))
    plt.imshow(canvas_color, cmap='gray')
    plt.axis('off')

    for pt0, pt1, score in matches:
        x0, y0 = pt0
        x1, y1 = pt1
        x1 += w  # shift x1 to the right image

        color = np.random.rand(3,)
        plt.plot([x0, x1], [y0, y1], color=color, linewidth=0.5)
        plt.scatter([x0, x1], [y0, y1], color=color, s=2)

    plt.title(f"Matched keypoints: {len(matches)}")
    plt.savefig(save_path, dpi=200)
    plt.show()

# Run example
if __name__ == "__main__":
    path0 = "./Datasets/SPUS/DIFF/S1602/imgs/00114.jpg"
    path1 = "./Datasets/SPUS/DIFF/S1704/imgs/00011.jpg"
    model_path = "./checkpoints/superpoint_lightglue_pipeline.ort.onnx"

    matches, img0_vis, img1_vis = run_lightglue_onnx(path0, path1, model_path)
    draw_matches(img0_vis, img1_vis, matches)
