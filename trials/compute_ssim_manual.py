import os
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def load_grayscale_image(path):
    return np.array(Image.open(path).convert('L'))

def compute_ssim(img1, img2):
    if img1.shape != img2.shape:
        img2 = np.array(Image.fromarray(img2).resize(img1.shape[::-1], Image.BILINEAR))
    return ssim(img1, img2)

def get_image_paths_from_txt(folder, txt_path):
    with open(txt_path, 'r') as f:
        filenames = [line.strip() for line in f if line.strip()]
    return [(fname, os.path.join(folder, fname)) for fname in filenames]

def get_image_paths(folder):
    supported_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    return [(f, os.path.join(folder, f)) for f in sorted(os.listdir(folder)) if os.path.splitext(f)[1].lower() in supported_exts]

def main(folder1, anchor_txt, folder2):
    anchors = get_image_paths_from_txt(folder1, anchor_txt)
    targets = get_image_paths(folder2)

    results = []

    for anchor_name, anchor_path in tqdm(anchors, desc="Comparing anchors"):
        results=[]
        img1 = load_grayscale_image(anchor_path)
        for target_name, target_path in targets:
            img2 = load_grayscale_image(target_path)
            score = compute_ssim(img1, img2)
            results.append((anchor_name, target_name, score))

        # 按 SSIM 从高到低排序
        results.sort(key=lambda x: x[2], reverse=True)

        # 打印结果
        print("source,target,ssim")
        for anchor, target, score in results[:10]:
            print(f"{anchor},{target},{score:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare SSIM between anchor images and a folder, sorted by SSIM.")
    parser.add_argument("folder1", type=str, help="Path to anchor image folder")
    parser.add_argument("anchor_txt", type=str, help="Path to anchor.txt")
    parser.add_argument("folder2", type=str, help="Path to second folder")
    args = parser.parse_args()

    main(args.folder1, args.anchor_txt, args.folder2)
