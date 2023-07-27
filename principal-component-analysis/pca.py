import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create a synthetic dataset with 3 features and 2 clusters
np.random.seed(42)

# Number of data points in each cluster
n_samples_per_cluster = 100

# Mean and standard deviation for each feature in the two clusters
# Mean_cluster defines the average values of the three components of the dataset
mean_cluster1 = [13, 5, 8]
# std_cluster defines the standard deviation of each feature in the cluster
std_cluster1 = [1, 1.5, 2]

mean_cluster2 = [6, 9, 12]
std_cluster2 = [1, 1.5, 2]


# Generate data points for each cluster with the number of features corresponding to len() of each cluster
cluster1_data = np.random.normal(loc=mean_cluster1, scale=std_cluster1, size=(n_samples_per_cluster, len(mean_cluster1)))
cluster2_data = np.random.normal(loc=mean_cluster2, scale=std_cluster2, size=(n_samples_per_cluster, len(mean_cluster2)))

# Combine the two clusters into a single dataset
X = np.vstack((cluster1_data, cluster2_data))

# Create labels for the clusters (0 for cluster1 and 1 for cluster2)
y = np.hstack((np.zeros(n_samples_per_cluster), np.ones(n_samples_per_cluster)))

# Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components='mle')
principal_components = pca.fit_transform(X_std)

# Create a DataFrame for the principal components
pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
pca_df["Cluster"] = y

# Plot the principal components
plt.figure(figsize=(8, 6))
plt.scatter(pca_df["PC1"], pca_df["PC2"], c=pca_df["Cluster"], cmap="viridis", edgecolors="k")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Analysis of Synthetic Dataset")
plt.colorbar(label="Cluster")
plt.show()

# Print explained variance ratio to see how much variance is explained by each component
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)
