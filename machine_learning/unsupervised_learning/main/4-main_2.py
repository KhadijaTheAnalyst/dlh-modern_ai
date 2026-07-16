#!/usr/bin/env python3

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from scipy.cluster.hierarchy import linkage, dendrogram

Standardize = __import__('0-standardize').Standardize
Apply_PCA = __import__('1-pca').Apply_PCA


X, _ = load_wine(return_X_y=True)
X_scaled = Standardize(X)

X_pca, _ = Apply_PCA(X_scaled, n_components=5, random_state=2)

Z_pca = linkage(X_pca, method='ward')
Z_original = linkage(X_scaled, method='ward')


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

dendrogram(
    Z_pca,
    truncate_mode='level',
    p=20,
    leaf_rotation=90.,
    leaf_font_size=10.,
    ax=axes[0],
    color_threshold=None
)
axes[0].set_title("Dendrogram - PCA-reduced data")
axes[0].set_xlabel("Samples")
axes[0].set_ylabel("Distance")
axes[0].grid(False)
axes[0].set_facecolor('white')

dendrogram(
    Z_original,
    truncate_mode='level',
    p=20,
    leaf_rotation=90.,
    leaf_font_size=10.,
    ax=axes[1],
    color_threshold=None
)
axes[1].set_title("Dendrogram - Original data")
axes[1].set_xlabel("Samples")
axes[1].set_ylabel("Distance")
axes[1].grid(False)
axes[1].set_facecolor('white')

plt.tight_layout()
plt.show()
