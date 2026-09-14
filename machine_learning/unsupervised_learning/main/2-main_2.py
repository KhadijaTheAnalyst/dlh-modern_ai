#!/usr/bin/env python3

from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Standardize = __import__('0-standardize').Standardize
K_Means = __import__('2-k_means').K_Means
Apply_PCA = __import__('1-pca').Apply_PCA



X, _ = load_wine(return_X_y=True)

X_scaled = Standardize(X)

X_3d, _ = Apply_PCA(X_scaled, n_components=3, random_state=2)

model = K_Means(X_scaled, n_clusters=3, random_state=2)
labels = model.labels_

fig = plt.figure(figsize=(10, 15))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
    c=labels, cmap='viridis', s=60, depthshade=True
)

ax.set_title("K-Means Clusters in 3D PCA Space")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")

legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

plt.tight_layout()
plt.show()
