The goal of this code is to implement a PCA analysis on a synthetic dataset. PCA is a preprocessing step to perform on a dataset before 
using a machine learning algorithm. It is a form of dimension reduction.

This python code generates a synthetic dataset with two clusters, each containing 3 features.
We then combine the two clusters into a single dataset and standardrize it with StandardScaler to put all the features on the same scale.
We then perform Principal Component Analysis (PCA) on the synthetic data with n_components='mle' (mle=Maximum Likelihood Estimation). 
Passing mle gives us the optimal number of principal components to retain based on the data. 

We store the principal components in the pca_df DataFrame and we add cluster labels for visualization purposes

In the plotting portion of the code starting on line 51, we have an conditional statement that plots 
based on the number of components (num_components_to_plot) since we don't know ahead of time what mle will return.

The PCA plot shows how the data points are distributed in the reduced 2D space of the first two principal components.


About:
PCA identifies principal components which capture the maximum variance in a dataset.
PCA is useful in dimension reduction. When dealing with data with many dimensions (features), it can be helpful to reduce the number of dimensions
which can throw off machine learning algorithms through overfitting. 

PCA can be used to visualize data, usually in 2d or 3d, to understand the data's structure and patterns. 
It can also be used for feature extraction - meaning it creates new features (principal componenets) that are linear combinations
of the original features that capture variability in the data. This helps remove noise from the data and retain only the most important dimensions.
In essence, we are taking high dimension data and reducing it a lower dimensional space. 

The first principal component accounts for the most significant variance, the second principal component has the second most significant variance, and so on.
By maximizing variance in the principal components, we reduce dimensionality of the data while still retaining the most critical information (the idea
being that high variance features are the most informative).
