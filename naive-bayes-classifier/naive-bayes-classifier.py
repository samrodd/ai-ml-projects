import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load the CSV file
def load_csv_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Train the Naive Bayes classifier
def train_naive_bayes(X_train, y_train):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    return clf

# Main function
def main():
    # Load data from CSV
    file_path = input("Enter the file path of the CSV data: ")
    data = load_csv_data(file_path)

    # Split data into features (X) and target variable (y)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Naive Bayes classifier
    classifier = train_naive_bayes(X_train, y_train)

    # Request new data points for classification
    new_data_points = []
    for i in range(2):
        feature_name = X.columns[i]
        value = input(f"Enter the value for {feature_name}: ")
        new_data_points.append(float(value))

    # Reshape the new_data_points to a 2-dimensional array
    new_data_points = [new_data_points]

    # Classify the new data points
    prediction = classifier.predict(new_data_points)

    print("Predicted class label:")
    print(prediction[0])

if __name__ == "__main__":
    main()
