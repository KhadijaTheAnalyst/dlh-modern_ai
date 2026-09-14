# Unsupervised Learning — Tasks Documentation

Repository: `dlh-modern_ai`
Directory: `machine_learning/unsupervised_learning`

All tasks use the Wine dataset (`sklearn.datasets.load_wine`) as the running example: 178 samples, 13 chemical features, standing in for 3 real wine cultivars (though the label is never given to the algorithms).

---

## Task 0 — Feature Standardization

**File:** `0-standardize.py`

**Goal:** Rescale every feature to mean 0, standard deviation 1, so no single feature dominates distance-based calculations purely because of its raw numeric scale.

**Formula:** `z = (x - mean) / std`, applied per feature/column.

```python
#!/usr/bin/env python3
"""Feature standardization module."""

from sklearn import preprocessing


def Standardize(X):
    """Standardize tabular data using Scikit-learn.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).

    Returns:
        numpy.ndarray: The standardized version of the input data, with the
            same shape as X.
    """
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled
```

**Verification:**
```
Original shape: (178, 13)
Standardized shape: (178, 13)
Mean per feature ≈ 0
Std per feature ≈ 1
```

---

## Task 1 — Dimensionality Reduction with PCA

**File:** `1-pca.py`

**Goal:** Project data onto fewer, uncorrelated axes (principal components) ordered by how much variance each one explains. `n_components` can be an `int` (exact count), a `float` between 0–1 (minimum variance fraction to preserve), or `None` (keep all components) — sklearn's `PCA` class handles all three natively.

```python
#!/usr/bin/env python3
"""Dimensionality reduction with PCA module."""

from sklearn import decomposition


def Apply_PCA(X, n_components, random_state):
    """Perform Principal Component Analysis (PCA) on tabular data.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_components (int, float, or None):
            - int: number of principal components to keep.
            - float (between 0 and 1): minimum fraction of variance
              to preserve.
            - None: keep all components.
        random_state (int): Random seed for reproducibility.

    Returns:
        numpy.ndarray: Data transformed into principal component space.
        sklearn.decomposition.PCA: Fitted PCA instance.
    """
    pca = decomposition.PCA(
        n_components=n_components,
        random_state=random_state
    )
    X_pca = pca.fit_transform(X)

    return X_pca, pca
```

**Verification:**
```
n_components=None -> shape (178, 13)
  explained_variance_ratio_: [0.362, 0.192, 0.111, 0.071, 0.066, 0.049,
                               0.042, 0.027, 0.022, 0.019, 0.017, 0.013, 0.008]

n_components=0.8  -> shape (178, 5)   (5 components needed to reach 80% variance)
n_components=5    -> shape (178, 5)
```

---

## Task 2 — Clustering with K-Means

**File:** `2-k_means.py`

**Goal:** Fit a K-Means model that partitions data into `n_clusters` groups by iteratively assigning points to the nearest centroid and recomputing centroids as the mean of their members.

```python
#!/usr/bin/env python3
"""Clustering with K-Means module."""

from sklearn import cluster


def K_Means(X, n_clusters, random_state):
    """Create and fit a K-Means clustering model on tabular data.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_clusters (int): Number of clusters to find.
        random_state (int): Random seed for reproducibility.

    Returns:
        sklearn.cluster.KMeans: K-Means instance fitted on the input data.
    """
    model = cluster.KMeans(
        n_clusters=n_clusters,
        random_state=random_state
    )
    model.fit(X)

    return model
```

**Verification:**
```
model.labels_.shape == (178,)
set(model.labels_) == {0, 1, 2}   (for n_clusters=3)
```

**Key insight demonstrated in 2-main_1.py:** running K-Means on raw (unstandardized) data vs. standardized data produces visibly different clusters when plotted in 2D PCA space — proof that standardization (Task 0) directly affects clustering quality, since K-Means relies entirely on distance.

---

## Task 3 — Choosing the Optimal K for K-Means

**File:** `3-optimal_k.py`

**Goal:** Evaluate K-Means across a range of `k` values (2 up to `max_clusters`) and collect two quality signals: **inertia** (for the elbow method) and **silhouette score** (for cluster quality). Reuses `K_Means` from Task 2.

```python
#!/usr/bin/env python3
"""Choosing the optimal k for K-Means module."""

from sklearn import metrics

K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """Evaluate K-Means clustering quality across a range of k values.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        max_clusters (int): Maximum number of clusters to evaluate (>= 2).
        random_state (int): Random seed for reproducibility.

    Returns:
        list[int]: Evaluated cluster numbers.
        list[float]: Inertia values corresponding to each cluster number,
            for the elbow method.
        list[float]: Silhouette scores corresponding to each cluster
            number, for cluster quality evaluation.
    """
    ks = []
    inertia_values = []
    silhouette_values = []

    for k in range(2, max_clusters + 1):
        model = K_Means(X, n_clusters=k, random_state=random_state)

        ks.append(k)
        inertia_values.append(model.inertia_)
        silhouette_values.append(
            metrics.silhouette_score(X, model.labels_)
        )

    return ks, inertia_values, silhouette_values
```

**Verification (max_clusters=10, standardized wine data):**
```
ks:          [2, 3, 4, 5, 6, 7, 8, 9, 10]
inertia:     [1659.95, 1277.93, 1187.11, 1105.66, 1043.81, 1002.43, 944.45, 903.88, 871.76]
silhouette:  [0.260, 0.285, 0.232, 0.229, 0.194, 0.197, 0.203, 0.135, 0.133]
```

Notice the silhouette score **peaks at k=3** — which matches the real (hidden) number of wine cultivars in the dataset. A nice confirmation that the metric works as intended.

Why the loop starts at `k=2`: silhouette score is mathematically undefined for `k=1` (there's no second cluster to measure separation against).

---

## Task 4 — Agglomerative Hierarchical Clustering

**File:** `4-agglomerative.py`

**Goal:** Fit Agglomerative (bottom-up) hierarchical clustering with Ward linkage, optionally on PCA-reduced data, and compute silhouette score.

```python
#!/usr/bin/env python3
"""Agglomerative hierarchical clustering module."""

from sklearn import cluster
from sklearn import metrics

Apply_PCA = __import__('1-pca').Apply_PCA


def Agglomerative_Clustering(X, n_clusters, random_state, n_components,
                             use_pca_data=True):
    """Perform Agglomerative hierarchical clustering on tabular data.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_clusters (int): Number of clusters.
        random_state (int): Random seed for reproducibility (used by PCA).
        n_components (int): Number of PCA components to reduce the data
            to (used only if use_pca_data=True).
        use_pca_data (bool): Whether to apply PCA to reduce
            dimensionality before clustering. Default is True.

    Returns:
        sklearn.cluster.AgglomerativeClustering: Fitted Agglomerative
            Clustering instance.
        numpy.ndarray: Data used for fitting (PCA-reduced or original).
        float: Silhouette score of the clustering (None if n_clusters=1).
    """
    if use_pca_data:
        X_used, _ = Apply_PCA(X, n_components, random_state)
    else:
        X_used = X

    model = cluster.AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    model.fit(X_used)

    if n_clusters > 1:
        score = metrics.silhouette_score(X_used, model.labels_)
    else:
        score = None

    return model, X_used, score
```

**Verification (standardized wine data, n_components=5):**
```
k=2, PCA  -> shape (178, 5), silhouette 0.323
k=2, orig -> shape (178, 13), silhouette 0.267
k=3, PCA  -> shape (178, 5), silhouette 0.347
k=3, orig -> shape (178, 13), silhouette 0.277
k=4, PCA  -> shape (178, 5), silhouette 0.291
k=4, orig -> shape (178, 13), silhouette 0.226
k=1, PCA  -> silhouette None (undefined for a single cluster)
```

**Key insight demonstrated in 4-main_1.py and 4-main_2.py:** PCA-reduced data consistently scores higher on silhouette than the original 13-dimensional data for this dataset — an example of dimensionality reduction improving cluster separability by discarding noisy/low-variance dimensions. The dendrogram (`4-main_2.py`, using `scipy.cluster.hierarchy.linkage` with `method='ward'`) visualizes the same merge process that `AgglomerativeClustering` performs internally.

**Note:** `AgglomerativeClustering` has no `random_state` parameter — the merging process is fully deterministic. `random_state` in this function is only used when `use_pca_data=True`, passed through to `Apply_PCA`.

---

## Pipeline Summary

The five tasks chain together into a full unsupervised learning workflow:

```
Raw data (X)
    │
    ▼
0. Standardize(X)            → X_scaled  (mean 0, std 1 per feature)
    │
    ├──────────────────────────────────────┐
    ▼                                       ▼
1. Apply_PCA(X_scaled, ...)             2. K_Means(X_scaled, ...)
    → X_pca, pca_model                      → fitted model
    │                                       │
    ▼                                       ▼
4. Agglomerative_Clustering(...)        3. optimal_k(X_scaled, ...)
    → model, X_used, silhouette score       → ks, inertia_values, silhouette_values
                                             (used to choose the best k for step 2)
```

- **Task 0** feeds every downstream task — never cluster or apply PCA on raw, unstandardized data.
- **Task 1** (PCA) is used both as a preprocessing step before clustering (Task 4) and independently, for visualizing high-dimensional clusters in 2D/3D.
- **Tasks 2 and 3** work together: `optimal_k` runs `K_Means` repeatedly to help choose a good `n_clusters` before committing to a final model.
- **Task 4** offers an alternative to K-Means that doesn't require randomly-initialized centroids and produces an interpretable dendrogram, with the option to cluster on PCA-reduced or original standardized data.
