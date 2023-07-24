This directory consists of a python file to run a Euclidian K Nearest Neighbors algorithm on a dataset of 100,000 loan applications. 

Code employs KNeighborsClassifier from sklearn.neighbors and train_test_split from sklearn.model_selection to split the dataset into 
training and testing data and then train the model.

The dataset contains following feature variables:

1. Loan_Amount (any integer)
2. FICO_SCORE (any integer 302-850 inclusive
3. Employment_Status (1 = unemployed; 2 = part_time; 3 = full_time)
4. Monthly_Gross_Income (any integer)
5. Monthly_Housing_Payment (any integer) 
6. Ever_Bankrupt_or_Foreclose (0 = false; 1 = true)
7. Lender (1-3 inclusive)

And the target variable is:
Approved (0 = false; 1 = true)

The goal of the code is to train a K Nearest Neighbors classifier using the training data and then request user input for a new loan application
and predict whether it will be approved or rejected.

In order to implement this accurately, I needed to split the training data based on the lender. Since different lenders have different
criteria for loan approval, not splitting the training data for each lender led to inaccurate results because the algorithm would consider
applications to irrelevant lenders in evaluating the nearest neighbors. So, the code takes in a lender value from the user and then trains
the classifier based on data from the provided lender. Then, it request new loan data from the user and outputs the predicted result of
the application based on the nearest 5 neighbors for that application.

Run k-nearest-neighbors.py and then input the lender you want to evaluate a new application for. 
Then, provide the feature variables above for the new loan application and see the output of the classifier.

**Reducing the feature variable list**
The original dataset consisted of additional feature variables like Employment_Sector, FICO_Score_Group, Loan_Purpose, and so on.
In testing the algorithm, I learned that these additional feature variables introduced too much noise such that the output did not
accurately predict the results of approved loans from the training data. Having too much data, as I did here, is known as the 
Curse of Dimensionality (the distance between data points loses its meaning).

Through iteration and testing the accuracy of the classifier on different sets of feature variables, 
I determined that the best 7 feature variables were the ones listed above.

**Determining K**
While there are 100,000 rows of data, each lender only had a fraction of the total rows, which meant there dataset was smaller.
With smaller datasets, it's helpful to have a smaller value of K. I tested values of 10, 7, 5, 3 and 1. The classifier performed
best when K=5.
