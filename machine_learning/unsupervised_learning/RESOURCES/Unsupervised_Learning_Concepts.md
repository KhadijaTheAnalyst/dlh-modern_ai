# Unsupervised Learning — Concepts Guide

A plain-language explainer for every learning objective in this project. Each section is written so you could explain it to someone else without needing to look anything up.

---

## 1. What is unsupervised learning?

Unsupervised learning is a type of machine learning where you give the algorithm data **without labels** — there's no "correct answer" attached to each row. The algorithm's job is to find structure, patterns, or groupings in the data on its own.

Example: hand an algorithm 178 wine samples with 13 chemical measurements each, but *no* information about which of the 3 wine cultivars each sample came from. Unsupervised learning tries to find natural groupings in that data — ideally, the groupings it finds line up with the real cultivars, even though it was never told they exist.

## 2. How does unsupervised learning differ from supervised learning?

| | Supervised Learning | Unsupervised Learning |
|---|---|---|
| Data | Features **+ labels** (e.g. "this wine is cultivar 2") | Features only, no labels |
| Goal | Learn to predict the label for new data | Discover structure/patterns in the data |
| Example task | Classification, regression | Clustering, dimensionality reduction |
| "Correctness" | Measurable against known answers | Judged by internal quality metrics (e.g. silhouette score), since there's no ground truth |

The key distinction: supervised learning has something to check its answers against. Unsupervised learning doesn't — it has to define its own notion of "good structure."

## 3. Why is it important to standardize data before applying clustering algorithms?

Clustering algorithms (K-Means, Agglomerative Clustering) group points based on **distance**. If your features are on wildly different scales — e.g. `proline` ranging 300–1680 vs. `hue` ranging 0.4–1.7 — the feature with the larger raw numbers will dominate every distance calculation, regardless of whether it's actually the most meaningful feature.

Standardizing rescales every feature to mean 0, standard deviation 1, using:

```
z = (x - mean) / std
```

This puts every feature on equal footing, so clustering results reflect genuine patterns in the data rather than accidents of measurement units.

## 4. What is dimensionality reduction and why is it useful?

Dimensionality reduction means taking data with many features (dimensions) and representing it with fewer features, while trying to preserve as much of the meaningful information as possible.

Why it's useful:
- **Visualization** — humans can't visualize 13 dimensions, but we can plot 2 or 3.
- **Noise reduction** — later, less-informative dimensions often carry mostly noise; dropping them can make clustering cleaner.
- **Efficiency** — fewer dimensions means faster computation for algorithms sensitive to feature count.
- **Removing redundancy** — correlated features carry overlapping information; reduction can consolidate that into fewer, more informative axes.

## 5. What is PCA and how does it help with dimensionality reduction?

PCA (Principal Component Analysis) is the most common dimensionality reduction technique. Instead of keeping your original features, PCA finds new axes — **principal components** — that are:

- Linear combinations of your original features
- Ordered by how much variance (spread) in the data they capture
- Mutually uncorrelated (each one captures information the others don't)

Component 1 captures the most variance possible along a single direction; Component 2 captures the most *remaining* variance, at a right angle to Component 1; and so on. You can then keep just the first few components and discard the rest, losing only the least informative parts of the data.

## 6. What is explained variance in PCA and why does it matter?

Explained variance ratio tells you what fraction of the total variance in your dataset each principal component accounts for. For the standardized wine dataset:

```
Component 1: 36.2%
Component 2: 19.2%
Component 3: 11.1%
...
```

This matters because it tells you how much information you're keeping (or throwing away) when you reduce dimensions. If the first 5 components together explain 80% of the variance, you can represent the dataset with 5 numbers per sample instead of 13, while retaining most of the meaningful structure. The **cumulative** explained variance (running total across components) is what you check against a target threshold, like "keep enough components to reach 80%."

## 7. What is K-Means clustering and how does it work?

K-Means partitions data into `k` clusters through an iterative process:

1. Place `k` centroids randomly in the feature space.
2. **Assign** every point to its nearest centroid — this becomes that point's cluster label.
3. **Update** each centroid to the mean position of all points currently assigned to it.
4. Repeat steps 2–3 until centroids stop moving significantly.

You must choose `k` in advance — K-Means doesn't discover the "right" number of clusters on its own.

## 8. What are cluster centroids?

A centroid is the "center point" of a cluster — literally the average (mean) position of all data points currently assigned to that cluster, computed separately for each feature/dimension. Centroids aren't real data points; they're computed averages that represent the "typical" member of a cluster. K-Means moves these centroids each iteration until they stabilize.

## 9. What is the Elbow Method and what is it used for?

The Elbow Method helps you pick a reasonable value of `k` for K-Means by looking at **inertia** — the sum of squared distances from each point to its assigned centroid, which measures how tightly packed the clusters are.

Inertia always decreases as `k` increases (more clusters = smaller, tighter groups), so you can't just pick the k with the lowest inertia. Instead, you plot inertia against k and look for the "elbow" — the point where the curve bends and adding more clusters stops producing large improvements. That bend suggests a good trade-off between cluster tightness and model simplicity.

## 10. How do you evaluate the quality of clusters?

Since there are no ground-truth labels in unsupervised learning, cluster quality is judged with **internal metrics** that only look at the data and the resulting groupings:

- **Inertia** — measures cohesion only (how tight clusters are); doesn't measure separation, and always improves with more clusters, so it's only useful as a relative "elbow" signal, not an absolute score.
- **Silhouette Score** — measures both cohesion *and* separation together, giving a single interpretable number that doesn't automatically favor more clusters.

## 11. What does the Silhouette Score indicate about clusters?

For each point, the Silhouette Score compares:
- How close that point is to others in its **own** cluster (cohesion)
- How close that point is to points in the **nearest other** cluster (separation)

The score ranges from -1 to 1:
- **Close to 1** — the point sits comfortably inside its own cluster, far from other clusters. Good.
- **Close to 0** — the point is right on the border between two clusters.
- **Negative** — the point is probably assigned to the wrong cluster.

Averaged across all points, this gives one score per value of `k`. Unlike inertia, the k with the **highest** silhouette score is a meaningful candidate for "best" number of clusters.

## 12. What is hierarchical (Agglomerative) clustering?

Agglomerative clustering builds clusters from the bottom up:

1. Start with every point as its own cluster.
2. Find the two closest clusters and merge them.
3. Repeat step 2 — always merging the closest pair — until you reach the desired number of clusters.

Unlike K-Means, it doesn't require randomly initialized centroids, and it naturally produces a full hierarchy of how points and groups merged together, from individual points all the way up to one giant cluster.

## 13. What is a dendrogram and how can it help interpret clusters?

A dendrogram is a tree diagram that visualizes the entire merge history from Agglomerative clustering. Each leaf is a single data point; each branch shows where two clusters merged, and the **height** of that branch shows how "far apart" (dissimilar) those clusters were when merged.

Reading a dendrogram helps you:
- See natural groupings by looking for clusters that merge at low heights (very similar) before merging with anything else at a much greater height.
- Choose a number of clusters by drawing a horizontal cut line — the number of vertical lines it crosses is the number of clusters at that "distance" threshold.
- Understand cluster relationships hierarchically, e.g. which sub-clusters are more similar to each other than to a third group.

## 14. What are linkage methods in hierarchical clustering?

When merging clusters (not just single points), you need a rule for measuring "distance between two clusters" (each of which may contain many points). That rule is the **linkage method**. Common options:

- **Ward linkage** — merges whichever pair of clusters causes the smallest increase in total within-cluster variance. Tends to produce compact, evenly-sized clusters. (Used in this project.)
- **Single linkage** — distance between the closest pair of points, one from each cluster. Can produce long, "chained" clusters.
- **Complete linkage** — distance between the farthest pair of points, one from each cluster. Tends to produce tight, compact clusters.
- **Average linkage** — average distance between all pairs of points across the two clusters.

The linkage method directly shapes the dendrogram and the resulting cluster boundaries — different linkage methods can produce noticeably different groupings from the same data.

## 15. How can you visualize clusters in reduced dimensions?

Since most real datasets have more than 2–3 features, you can't directly plot the clusters. The common workaround:

1. Reduce the data to 2 or 3 dimensions with PCA.
2. Plot the PCA-reduced points, colored by their cluster label.
3. Optionally, transform the cluster centroids into the same PCA space (using the same fitted PCA model) and overlay them on the plot.

This is exactly what `2-main_1.py` and `4-main_1.py` do in this project — they cluster the data (in either original or PCA space), then always plot using 2D or 3D PCA coordinates, since that's the only way to see the groupings visually.

## 16. How can dimensionality reduction affect clustering results?

Reducing dimensions before clustering can change results, sometimes significantly:

- **Can help**: PCA removes noisy, low-variance dimensions and correlated redundancy, which can make distance calculations more meaningful and clustering more stable — especially useful in high-dimensional data where distances between points become less meaningful ("curse of dimensionality").
- **Can hurt**: if you reduce too aggressively, you may discard dimensions that actually contained meaningful separating information, causing clusters to blur together that were previously distinct.

This project demonstrates the trade-off directly: `4-main_1.py` runs Agglomerative Clustering on both PCA-reduced and original standardized data, showing that silhouette scores and cluster shapes can differ between the two — there's no universal rule that one is always better; it depends on the dataset.
