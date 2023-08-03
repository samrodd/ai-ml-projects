# Loughlin Rodd
# Loan Bounty Optimization Using Classification and Regression Trees Algorithm (CART)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Step 1 - preprocess the dataset, split it into training and test data sets, and establish a baseline (approval rate)
# for each lender 
# Load the data from a CSV
df = pd.read_csv('/Users/loughlinrodd/Desktop/ai-problems/loan-bounty-optimization/loan_data.csv')

# Define the bounty for each lender
bounties = {0: 250, 1: 350, 2: 150}

# Step 2: Calculate the baseline approval rates
baseline_approval_rates = df.groupby('lender')['Approved'].mean()
print("Baseline Approval Rates:")
print(baseline_approval_rates)

# Step 3: Calculate the total of the bounty column by summing the product of the bounty for each lender and each approval
total_bounty_before = (df['Approved'] * df['lender'].map(bounties)).sum()
print(f"Total Bounty Before Decision Tree: {total_bounty_before}")

# Define a dictionary to store the total bounty per lender
total_bounty_per_lender_before = {0: 0, 1: 0, 2: 0}

# Iterate through each lender and calculate the total bounty
for lender in bounties.keys():
    # Filter the dataframe df to include only the rows for the lender in question where the loan is approved and access the number of rows with .shape[0]
    approved_loans = df[(df['lender'] == lender) & (df['Approved'] == 1)].shape[0]
    # Multiply the approved_loans by the bounty for the lender in question 
    total_bounty_per_lender_before[lender] = approved_loans * bounties[lender]

# Print the total bounty per lender before optimization
for lender, bounty in total_bounty_per_lender_before.items():
    print(f"Total bounty for lender {lender}: {bounty}")




# Step 4: Preprocess data for a decision tree - define the feature columns
feature_cols = ['Loan_Amount', 'FICO_SCORE', 'Employment_Status', 'Monthly_Gross_Income', 'Monthly_Housing_Payment', 'Ever_Bankrupt_or_Foreclose']
# extract feature_cols into dataframe X
X = df[feature_cols]
# extract the 'Approved' column into dataframe Y
y = df['Approved']

# Step 5: Create a decision tree for each lender and print details
lender_models = {}
# Iterate through each lender 
for lender in df['lender'].unique():
    # Filter the dataframe df so that only the rows for the rows for the current lender are held in lender_data
    lender_data = df[df['lender'] == lender]
    # Extract the feature_cols from lender_data into X_lender
    X_lender = lender_data[feature_cols]
    # Extract the 'Approved' column from lender_data into y_lender
    y_lender = lender_data['Approved']
    # Split the data for the current lender into training and testing sets using 20% of the data for testing. 
    X_train, X_test, y_train, y_test = train_test_split(X_lender, y_lender, test_size=0.2, random_state=42)
    # Set clf to a decision tree classififer object with random_state=42 to set a seed for the random number generator used by the decision tree model. This will make our results deterministic
    # And uniform each time we run the code
    clf = DecisionTreeClassifier(random_state=42)
    # Fit clf to the training data
    clf.fit(X_train, y_train)
    # Print lender, depth of tree and number of leaves in tree
    print(f"Lender {lender}:")
    print(f"  Depth: {clf.tree_.max_depth}")
    print(f"  Number of leaves: {clf.tree_.n_leaves}")
    # Store the trained decision tree classifier in a dictionary with a key corresponding to that lender
    lender_models[lender] = clf

# Step 6: Optimize how applications are associated with lenders such that the total bounty is maximized
# Define variable for total bounty after optimizing applications
total_bounty_after = 0
# Define dictionary for number of approved loans for each lender after optimization
approved_loans_after = {0: 0, 1: 0, 2: 0}
# Define dictionary for total bounty coming from each lender
total_bounty_per_lender_after = {0: 0, 1: 0, 2: 0}

# Iterate through each row of the dataframe df
for index, row in df.iterrows():
    # Loop through the lenders in order of their bounty, descending
    for lender in [1, 0, 2]: 
        # Get the trained decision tree model for the current lender 
        model = lender_models[lender]
        # Use the model to predict the approval status of the loan in the current row of data by extracting the feature columns and feeding them to the model
        prediction = model.predict([row[feature_cols]])
        # Check if the loan is approved or not
        if prediction[0] == 1: # Loan approved
            # Increment total_bounty_after by the bounty of the approving lender
            total_bounty_after += bounties[lender]
            # Increment approved_loans_after for the approving lender
            approved_loans_after[lender] += 1
            # Incremenent the total_bounty_per_lender_after by the bounty amount for that lender
            total_bounty_per_lender_after[lender] += bounties[lender]
            break # Exit the loop if loan approved. Since loans are evaluated by highest bounty lender first, once we get an approval we can exit the loop.

# Step 6: Print the required information
print("Number of Approvals Before Decision Tree:")
print(baseline_approval_rates * df.groupby('lender').size())
print(f"Total Bounty Before Decision Tree: {total_bounty_before}")

# Print the number of approvals and total bounty after optimization
print("Number of Approvals After Decision Tree:")
print(approved_loans_after)
print(f"Total Bounty After Decision Tree: {total_bounty_after}")

# Print the total bounty per lender before optimization
for lender, bounty in total_bounty_per_lender_before.items():
    print(f"Total bounty for lender before optimization: {lender}: {bounty}")

# Print the total bounty per lender after optimization
for lender, bounty in total_bounty_per_lender_after.items():
    print(f"Total bounty for lender after optimization: {lender}: {bounty}")
