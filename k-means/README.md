This code runs a K-Means clustering algorithm from the sklearn.cluster package on the Iris dataset. 

K-Means is an unsupervised learning algorithm, meaning training data does not have a target variable. 
The algorithm clusters data points into distinct groups based on their similarity. The number of clusters (K) is predfined, in my code K=3.

The algorith has 4 steps: 
1. Initialization: Choose 3 centroids randomly from the data as centers of the clusters (each centroid is a point
that represents the center of a cluster. It is the average of all the data points for a given cluster).
2. Assignment: Assign each data point to the nearest centroid based on Euclidian distance, forming the 3 clusters.
3. Update: Recalculate the position of the 3 centroids as the average of all the data poitns assigned to its cluster.
4. Iteration: Repeat the assignment and update steps until convergence (when the centroids no longer change significantly)

The Iris dataset contains measurements of iris flowers with three distinct species. 
It has four numerical features: sepal length, sepal width, petal length, and petal width.

The algorithm clusters the data based on sepal length and sepal width measurements.
It first standardizes these features using StandardScaler from sklearn.preprocessing. 
This transforms the features so that each have a mean of 0 and standard deviation of 1. 

Standardization is a common step in preprocessing data for K-means to  ensures all features have qual importance and influence in the clustering process.
However, this process did cause the centroids to have different scales compared to the original data points.

I wanted the centroids to have the same scale as the datapoints so that the output visualization appeared more representative of the data.
To achieve this, I used the inverse_transform() method from StandardScaler to bring the centroids back to their original scale,
allowing me to see how they compare to the actual data points.
Prior to this, the centroids appeared far a way from the clusters they represent. After the inverse_transform() method, the centroids
appeared in the center of the clusters they represent.

The K-Means visualization shows how the Iris data points are grouped into clusters based similarity in the feature space. 
This gives us insights into how the algorithm partitions the data and assigns each data point to a particular cluster.

Run k-means.py to see the visualization - or view K-Means-Iris-Dataset-Figure_1.png.
