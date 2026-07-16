#!/usr/bin/env python3

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

Standardize = __import__('0-standardize').Standardize
optimal_k = __import__('3-optimal_k').optimal_k


X, _ = load_wine(return_X_y=True)
X_scaled = Standardize(X)

ks, inertia_values, silhouette_values = optimal_k(
    X_scaled, max_clusters=10, random_state=2
)

sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.lineplot(
    x=ks,
    y=inertia_values,
    marker="o",
    ax=axes[0]
)
axes[0].set_title("Elbow Method (Inertia)")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].set_xticks(ks)
axes[0].grid(True, linestyle="--", alpha=0.5)

sns.lineplot(
    x=ks,
    y=silhouette_values,
    marker="o",
    ax=axes[1]
)
axes[1].set_title("Silhouette Score vs Number of Clusters")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_xticks(ks)
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
