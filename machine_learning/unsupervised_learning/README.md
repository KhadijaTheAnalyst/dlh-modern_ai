# Unsupervised Learning

## Description

This project covers core unsupervised learning techniques using Scikit-learn: feature standardization, dimensionality reduction with PCA, and clustering with K-Means and Agglomerative Hierarchical Clustering. Each task builds on the ones before it — standardized data feeds into PCA, and both feed into the clustering algorithms.

The Wine dataset (`sklearn.datasets.load_wine`) is used throughout the example scripts to demonstrate each function.

## Requirements

* Python 3
* numpy
* pandas
* scikit-learn
* scipy
* seaborn
* matplotlib

## Files

| File | Description |
|---|---|
| `0-standardize.py` | `Standardize(X)` — standardizes tabular data (mean 0, std 1 per feature) using `sklearn.preprocessing.StandardScaler`. |
| `1-pca.py` | `Apply_PCA(X, n_components, random_state)` — reduces dimensionality with Principal Component Analysis. Supports `int`, `float` (variance ratio), or `None` for `n_components`. |
| `2-k_means.py` | `K_Means(X, n_clusters, random_state)` — fits a K-Means clustering model. |
| `3-optimal_k.py` | `optimal_k(X, max_clusters, random_state)` — evaluates K-Means across a range of `k` values using inertia (elbow method) and silhouette score (cluster quality). |
| `4-agglomerative.py` | `Agglomerative_Clustering(X, n_clusters, random_state, n_components, use_pca_data=True)` — fits Agglomerative Hierarchical Clustering (Ward linkage), optionally on PCA-reduced data, with silhouette score evaluation. |

## Usage

Each file can be imported and used independently, or chained together:

```python
Standardize = __import__('0-standardize').Standardize
Apply_PCA = __import__('1-pca').Apply_PCA
K_Means = __import__('2-k_means').K_Means
optimal_k = __import__('3-optimal_k').optimal_k
Agglomerative_Clustering = __import__('4-agglomerative').Agglomerative_Clustering

from sklearn.datasets import load_wine
X, _ = load_wine(return_X_y=True)

# 1. Standardize
X_scaled = Standardize(X)

# 2. Reduce dimensionality
X_pca, pca_model = Apply_PCA(X_scaled, n_components=0.8, random_state=2)

# 3. Cluster
model = K_Means(X_scaled, n_clusters=3, random_state=2)

# 4. Find the optimal number of clusters
ks, inertia_values, silhouette_values = optimal_k(X_scaled, max_clusters=10, random_state=2)

# 5. Try hierarchical clustering instead
agg_model, X_used, score = Agglomerative_Clustering(
    X_scaled, n_clusters=3, random_state=2, n_components=5, use_pca_data=True
)
```

## Key Concepts

* **Standardization** rescales each feature to mean 0, standard deviation 1, so features with different units/ranges contribute equally to distance-based calculations.
* **PCA** projects data onto a smaller set of uncorrelated axes (principal components) that capture the most variance, reducing dimensionality while retaining as much information as possible.
* **K-Means** partitions data into `k` clusters by iteratively assigning points to the nearest centroid and recomputing centroids as the mean of their assigned points.
* **Elbow method** (inertia) and **silhouette score** are two complementary ways to choose a good value of `k` — inertia always decreases with more clusters, so look for a bend in the curve; silhouette score peaks at a genuinely well-separated `k`.
* **Agglomerative Hierarchical Clustering** builds clusters bottom-up by repeatedly merging the closest pair of clusters (Ward linkage minimizes the increase in within-cluster variance at each merge), producing a hierarchy that can be visualized as a dendrogram.

## Author

Khadija Mustafa