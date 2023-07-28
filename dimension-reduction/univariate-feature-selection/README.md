The purpose of this code is to perform univariate feature selection on a dataset of loan applications to three separate lenders to determine
which feature variables are most important in determining the target variable (loan approval or rejection).

The python code will take in a csv of loan application data and reduce the 7 feature variables to the 4 most important ones for each lender
and print them in descending order of importance along with their ANOVA-F value. The F value represents an assessment of the significance of
the variable in determining the target variable. The higher the F value of the feature variable, the more important it is in determining the
target variable. 

The code uses the SelectKBest class from scikit-learn to select the top k (k=4) features and uses f_classif as the scoring function, 
which gives us the F value.

After trial and error, I decided to pick k=4 because I wanted to selection to output non-uniform features for each lender. With values of 1-3, 
the selection output the same variables for each lender. In essence, I wanted to find the point at which for at least one of the lenders the
set of the most important features differed in some way. At k=4, we can see that Lender 1 has a 4th most important feature of ever_bankrupt_or_foreclose,
while Lenders 2 and 3 have a 4th most important value of loan_amount. Prior to the 4th most important feature, 
each lender prioritized features of fico_score, monthly_gross_income and employment_status, in that order. 

We can see from the output of the code that the F values, or the degree to which the feature is important, differ amongst the three lenders, but the
ranking is still the same across all three until we get to feature 4.

Ranked features for 1: FICO_SCORE (F-value: 4148.94), Monthly_Gross_Income (F-value: 1955.28), Employment_Status (F-value: 158.70), Ever_Bankrupt_or_Foreclose (F-value: 85.90)
Ranked features for 2: FICO_SCORE (F-value: 3070.85), Monthly_Gross_Income (F-value: 1113.30), Employment_Status (F-value: 152.34), Loan_Amount (F-value: 30.26)
Ranked features for 3: FICO_SCORE (F-value: 1247.40), Monthly_Gross_Income (F-value: 473.18), Employment_Status (F-value: 428.16), Loan_Amount (F-value: 71.87)
