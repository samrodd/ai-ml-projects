import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate synthetic data with 7 clusters
X, y = make_blobs(n_samples=700, centers=7, random_state=42)

# Standardize the data (optional but can help improve clustering results)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Perform hierarchical clustering
agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
clusters = agg_clustering.fit_predict(X_std)

# Plot dendrogram (optional, for visualization)
linked = linkage(X_std, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title("Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# Print cluster assignments
print("Cluster Assignments:")
for i in range(len(X)):
    print(f"Sample {i}: Cluster {clusters[i]}")
