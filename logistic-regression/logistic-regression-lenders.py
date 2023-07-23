import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the CSV file as data
def load_csv_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Train the logistic regression model
def train_logistic_regression(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    clf = LogisticRegression()
    clf.fit(X_train_scaled, y_train)
    return clf, scaler

# Request new data points for classification
def classify_new_data(classifier, new_data_points, scaler):
    new_data_scaled = scaler.transform(new_data_points)
    prediction = classifier.predict(new_data_scaled)
    return prediction[0]

# Main function
def main():
    # Load data from CSV
    file_path = 'filepath'  # Replace with the actual file path
    data = load_csv_data(file_path)

    # Separate features and labels
    X = data.drop('Approved', axis=1)
    y = data['Approved']

    # Train the logistic regression model
    classifier, scaler = train_logistic_regression(X, y)

    # Request new data points for classification
    new_data_points = []
    for column in X.columns:
        value = float(input(f"Enter the value for {column}: "))
        new_data_points.append(value)

    # Convert the new data points to a numpy array and reshape for prediction
    new_data_points = np.array(new_data_points).reshape(1, -1)

    # Classify the new data points
    prediction = classify_new_data(classifier, new_data_points, scaler)
    if prediction == 1:
        print("Approved: True")
    else:
        print("Approved: False")

if __name__ == "__main__":
    main()
