The loan bounty optimization decision tree uses a Classification and Regression Tree (CART) algorithm from sklearn.tree. It is written in Python.

The goal of the optimization is to maximize the total bounty we are paid for submitting loan applications that are approved to one of three different lenders.
We take a pre-existing set of loan data containing loan applications to three different lenders (loan_data.csv)
and train a decision tree for each lender to predict whether new loans will be approved or rejected. 
The lenders pay us different bounties: Lender 0 pays $250. Lender 1 pays $350. Lender 2 pays $150.
Thus, we want to first maximize the number of approved loans from Lender 1, then Lender 0 then Lender 2. 

After training our decision trees for each lender, we first submit each row of data to be predicted by the model for Lender 1. 
If Lender 1 approves, we increment our bounty and move on to the next row. If the model predicts Lender 1 will reject the loan, we repeat 
the process for lenders 0 and 2. 

The code sets a seed for the random number generator used by the decision tree model. This will make our results deterministic
and uniform each time we run the code.

The code prints the total bounty before and after optimization, the total approval per lender before and after optimization, and the 
total bounty per lender before and after optimization.

By optimizing in this way, we can achieve a ~2.76 increase in total bounty after optimization.
