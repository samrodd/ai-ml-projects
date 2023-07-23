This code runs a logistic regression algorithm on a dataset of 100,000 loan applications. 

Code employs using LogisticRegression from Sklearn.linear_model and StandardScaler from Sklearn.preprocessing to standardize the dataset.

The dataset contains following data points:

1. applications (always 1)
2. reason (1 = cover_an_unexpected_cost; 2 = credit_card_refinancing; 3 = debt_consolidation; 4 = home_improvement; 5 = major_purchase; 6 = other)
3. Loan_Amount (any integer)
4. FICO_SCORE (any integer 302-850 inclusive
5. fico_score_group (1 = poor; 2 = fair; 3 = good; 4 = very_good; 5 = excellent)
6. Employment_Status (1 = unemployed; 2 = part_time; 3 = full_time)
7. Employment_Sector (1 = communication services; 2 = consumer_discretionary; 3 = consumer_staples; 4 = energy; 5 = financials; 6 = health_care; 7 = industrials; 8 = information technology; 9 = materials; 10 = real_estate; 11 = utilities; 12 = unemployed)
8. Monthly_Gross_Income (any integer)
9. Monthly_Housing_Payment (any integer) 
10. Ever_Bankrupt_or_Foreclose (0 = false; 1 = true)
11. Lender (1-3 inclusive) 
12. Approved (0 = false; 1 = true)

The goal of the code is to train a Logistic Regression on the dataset and then request user input for a new loan application and predict whether it will be approved or rejected.

Simply download the python file and the dataset to the same directory and run the python file. 

Provide the inputs when prompted and see whether 'approved' for the new loan application is 'true' or 'false'

About:
Logistic Regression is a supervised learning model used in predictive analysis. It works when the outcome of the classification is binary.
The purpose of Logistic Regression is to determine the relationship between independent input variables and the probability of the outcome being true of false.
If is essentially predicting whether the output is True or False based on the the input variables.
