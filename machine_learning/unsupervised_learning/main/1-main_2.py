#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
Standardize = __import__('0-standardize').Standardize
Apply_PCA = __import__('1-pca').Apply_PCA


X, _ = load_wine(return_X_y=True)

X_scaled = Standardize(X)

_, pca = Apply_PCA(X_scaled, n_components=None, random_state=2)

cumulative_var = np.cumsum(pca.explained_variance_ratio_) * 100  # convert to percent
components = np.arange(1, len(cumulative_var) + 1)

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(x=components, y=cumulative_var, marker="o")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("Cumulative Variance Explained by PCA Components")
plt.xticks(components)
plt.ylim(0, 105)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
