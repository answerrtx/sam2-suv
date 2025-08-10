"""
CLIP的feature
"""
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)

# 图像加载与预处理
path0="./Datasets/MSKUSO/shd9/imgs/00029.jpg"
image = Image.open(path0).convert("RGB")
img_pre = clip_preprocess(image).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

def get_clip_patch_features(image: Image.Image, model, device):
    # Preprocess
    image_input = clip_preprocess(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    with torch.no_grad():
        x = image_input  # shape: [1, 3, 224, 224]
        x = model.visual.conv1(x)  # shape: [1, 768, 16, 16]  (ViT-B/16)

        x = x.reshape(x.shape[0], x.shape[1], -1)  # [1, 768, 196]
        x = x.permute(0, 2, 1)  # [1, 196, 768]

        # Prep CLS token
        cls_token = model.visual.class_embedding.to(x.dtype)
        cls_token = cls_token + torch.zeros(x.shape[0], 1, cls_token.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)  # [1, 197, 768]

        x = x + model.visual.positional_embedding.to(x.dtype)
        x = model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # [197, 1, 768]
        for block in model.visual.transformer.resblocks:
            x = block(x)
        x = x.permute(1, 0, 2)  # [1, 197, 768]
        x = model.visual.ln_post(x)

        patch_features = x[:, 1:, :]  # remove CLS token -> [1, 196, 768]

    return patch_features.squeeze(0).cpu().numpy()  # shape: [196, 768]


# Hook 提取 patch 特征
patch_tokens = []

def hook_patch_features(module, input, output):
    # output 是 patch token（包括 CLS），取 1: 之后的是 patch tokens
    patch_tokens.append(output[:, 1:, :].detach())  # shape: [1, num_patches, dim]

handle = clip_model.visual.transformer.register_forward_hook(hook_patch_features)

# 前向传播
with torch.no_grad():
    _ = clip_model.encode_image(img_pre)

handle.remove()

# 得到 patch tokens
#features = patch_tokens[0]  # Tensor: [1, 196, 512]
clip_model = clip_model.float()

features = get_clip_patch_features(image, clip_model, device)  # shape: [196, 768]

# t-SNE 降维
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
tsne_result = tsne.fit_transform(features)  # [196, 2]

# 映射到 RGB 热图（14×14 patch-grid）
tsne_norm = (tsne_result - tsne_result.min(0)) / (tsne_result.max(0) - tsne_result.min(0))
tsne_rgb = tsne_norm.reshape(14, 14, 2)
tsne_rgb = np.concatenate([
    tsne_rgb,
    1 - np.linalg.norm(tsne_rgb - 0.5, axis=2, keepdims=True)
], axis=2)

# 可视化
plt.figure(figsize=(5, 5))
plt.imshow(tsne_rgb)
plt.title("CLIP Patch Feature t-SNE Grid")
plt.axis('off')
plt.tight_layout()
plt.show()
