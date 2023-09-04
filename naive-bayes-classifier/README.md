This code generates a dataset suitable to run a Naive Bayes Classifier on.

First, run generate-naive-bayes-dataset.py and replace the filepath variable with the directory your code lives in and a filename ending in .csv. 
This will generate a CSV with the dataset for the Naive Bayes Classifier. It will need numeric feature variables and target variable.

Second, run naive-bayes-classifier.py. It will request the filepath of the dataset created in step 1. It will then request two features. 
For the features, enter in positive or negative numeric decimal values. 
For example, you can use:
Feature 1: .30154734233361247
Feature 2: 2.3649610024662255

The code uses sklearn's train_test_split function to train the dataset and naive_bayes to import the Gaussian Naive Bayes classifier.

About:
Naive Bayes is a classifier algorithm that assumes each pair of features being classified is independent of each other. 

The dataset used for Naive Bayes consists of:
- a feature matrix containing the vectors of data (i.e., each row of data).
- a response vector showing the output value for each vector of the matrix (i.e., true or false)

In Naive Bayes, each feature makes an equal and independent contribution to the outcome. So, each feature is independent and given the same weigthing towards the outcome.

Even though features in the real world are often NOT independent, the Naive Bayes still works well in practice. 

