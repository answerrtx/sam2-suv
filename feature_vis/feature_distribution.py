import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict

# 输入路径：包含多个 .npy，每个是一个 dict，键是 label，值是 (768,)
features_dir = "./Datasets/MSKUSO/shoulder/shd2_features"

all_features = []
all_labels = []

# 读取每个特征文件
for fname in os.listdir(features_dir):
    if not fname.endswith(".npy"):
        continue
    data = np.load(os.path.join(features_dir, fname), allow_pickle=True).item()
    for label, feat in data.items():
        all_features.append(feat)
        all_labels.append(label)

all_features = np.stack(all_features)
print(f"Total features: {len(all_labels)}")

# 降维处理（可以换成 PCA）
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne.fit_transform(all_features)

# 给每个标签分配颜色
unique_labels = sorted(set(all_labels))
label_to_color = {label: plt.cm.tab20(i % 20) for i, label in enumerate(unique_labels)}

# 绘图
plt.figure(figsize=(10, 8))
for label in unique_labels:
    idx = [i for i, l in enumerate(all_labels) if l == label]
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=30, label=label, color=label_to_color[label], alpha=0.8)

plt.title("t-SNE of DINOv2 Features per Label")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.grid(True)
plt.show()
