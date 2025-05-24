# ssim_module.py

from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np

def load_image_as_grayscale(path):
    """Load image and convert to grayscale numpy array."""
    image = Image.open(path).convert('L')  # 'L' mode converts to grayscale
    return np.array(image)

def compute_ssim(img_path1, img_path2):
    """
    Compute SSIM between two images given their paths.
    
    Args:
        img_path1 (str): Path to first image.
        img_path2 (str): Path to second image.

    Returns:
        float: SSIM value between the two images.
    """
    img1 = load_image_as_grayscale(img_path1)
    img2 = load_image_as_grayscale(img_path2)

    # Ensure the shapes match
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes do not match: {img1.shape} vs {img2.shape}")

    ssim_value, _ = ssim(img1, img2, full=True)
    return ssim_value

if __name__ == "__main__":
    # Example usage
    img1_path = "path/to/image1.jpg"
    img2_path = "path/to/image2.jpg"
    
    ssim_value = compute_ssim(img1_path, img2_path)
    print(f"SSIM: {ssim_value}")