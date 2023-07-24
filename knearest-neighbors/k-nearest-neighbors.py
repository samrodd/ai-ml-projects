import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset from CSV file
dataset_path = '/Users/loughlinrodd/Desktop/ai-problems/knearest-neighbors/lender-dataset-nearest-neighbors.csv'
df = pd.read_csv(dataset_path)

# target variable column name is 'Approved', and 'lender' is one of the feature columns
X = df.drop(columns=['Approved'])
y = df['Approved']

# Prompt the user to enter the lender for which they want to predict the loan status
user_lender = int(input("Enter the lender (1, 2, or 3) for which you want to predict the loan status: "))

# Filter the dataset based on the specified lender
lender_data = df[df['lender'] == user_lender]
X_lender = lender_data.drop(columns=['Approved', 'lender'])
y_lender = lender_data['Approved']

# Split the lender-specific dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lender, y_lender, test_size=0.2, random_state=42)

# Create the K Nearest Neighbors classifier with k=5 and Euclidean distance
knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# Train the classifier using the lender-specific training data
knn_classifier.fit(X_train, y_train)

# Prompt the user to enter new feature values for a loan application
print("Please enter the following feature values for the loan application:")
new_loan_application = {}
for feature in X_lender.columns:  # Iterate over the filtered features for the selected lender
    value = float(input(f"{feature}: "))
    new_loan_application[feature] = value

# Convert the user's input into a DataFrame and make a prediction
new_application_df = pd.DataFrame([new_loan_application])
predicted_loan_status = knn_classifier.predict(new_application_df)

# Output the predicted target variable (loan approval status)
print("Predicted Loan Status:", predicted_loan_status[0])
