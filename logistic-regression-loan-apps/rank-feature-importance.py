import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the CSV file as data
def load_csv_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Main function
def main():
    # Load data from CSV
    file_path = 'filepath'  # Replace with the actual file path
    data = load_csv_data(file_path)

    # Split data into features (X) and target variable (y)
    X = data.drop('Approved', axis=1)
    y = data['Approved']

    # Train the logistic regression model
    clf = LogisticRegression()
    clf.fit(X, y)

    # Get the coefficients (importance) of the features
    feature_importance = dict(zip(X.columns, clf.coef_[0]))

    # Sort the features based on their importance in descending order
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)

    # Output the importance of the feature variables in descending order
    print("Feature Importance (Descending Order):")
    for feature, importance in sorted_features:
        print(f"{feature}: {importance}")

if __name__ == "__main__":
    main()
