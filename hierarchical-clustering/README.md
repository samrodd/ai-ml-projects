This code runs a hierarchical clustering algorithm in Python and outputs a dendogram.

The code generates a synthetic blob dataset with 7 clusters in line 9. 
However, we don't tell the AglomerativeClustering function (from sklearn.cluster) this in line 16, instead we set n_clusters=NONE. 
This ensures there is not a fixed number of clusters in the dendogram. 

We then run the clustering algorithm and output the dendogram and can see that it groups the datapoints into 7 distinct clusters at the bottom of the tree.
See the dendogram image in the repo for further inspection.

**About:**

In hierarchical clustering, data points are successively merged into clusters based on their similarity or distance,
creating a tree-like structure called a dendrogram. The dendrogram visually displays the hierarchical relationships between clusters, 
with shorter branches representing stronger similarities and longer branches indicating greater dissimilarity between clusters.

By looking at the dendogram we can learn about:
- cluster similarity (the height of a branch in the dendogram indicates the similarity/distance between the clusters being merged)
- number of clusters (shows natural clusters in the data)
- hierarchical relationships (shows which data points or clusters are merged at each level)
- Outliers (can show if there are any isolated branches or single data points that do not merge with other clusters)
- Imbalance issues (lengths of branches and number of data points per cluster can show potential imbalance issues. Imbalanced clusters can be represented by significantly shorter branches)
- Merging order (The order in which clusters are merged important since clusters that merge early in the process are probably more closely related than clusters that merge later) 
