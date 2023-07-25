import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset from scikit-learn
data = load_iris()
X = data.data
y = data.target

# Standardize the features for better clustering results
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to perform K-Means clustering
def kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters

# Perform K-Means clustering with 3 clusters (since there are 3 iris species in the dataset)
n_clusters = 3
clusters = kmeans_clustering(X_scaled, n_clusters)

# Create a DataFrame with the clustered data and add the species labels
clustered_df = pd.DataFrame(X, columns=data.feature_names)
clustered_df['Cluster'] = clusters
clustered_df['Species'] = data.target_names[y]

# Visualize the clusters in 2D using the first two principal components
plt.figure(figsize=(10, 8))
markers = ['o', 's', '^']  # Different markers for each cluster
for i in range(n_clusters):
    cluster_data = clustered_df[clustered_df['Cluster'] == i]
    plt.scatter(cluster_data['sepal length (cm)'], cluster_data['sepal width (cm)'],
                label=f'Cluster {i}', alpha=0.7, marker=markers[i], edgecolors='k')

# Create a new KMeans object and fit it to the data to access cluster centers
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)

# Transform the scaled centroids back to the original feature space
centroids_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)

# Plot cluster centroids with annotations in the original feature space
for i, centroid in enumerate(centroids_original_scale):
    plt.scatter(centroid[0], centroid[1], c='red', s=200, marker='X', label=f'Centroid {i}')
    plt.annotate(f'Centroid {i}', (centroid[0], centroid[1]), xytext=(centroid[0] + 0.15, centroid[1] + 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05))

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('K-Means Clustering of Iris Dataset')
plt.legend()
plt.grid()
plt.show()
