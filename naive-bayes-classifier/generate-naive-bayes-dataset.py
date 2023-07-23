import numpy as np
import pandas as pd

def generate_synthetic_dataset(num_samples=1000, random_seed=42):
    np.random.seed(random_seed)

    # Generate random feature values
    feature1 = np.random.normal(loc=0, scale=1, size=num_samples)
    feature2 = np.random.normal(loc=2, scale=1, size=num_samples)

    # Create target variable based on the features
    target = (feature1 + feature2 + np.random.normal(loc=0, scale=0.5, size=num_samples)) > 2

    # Combine features and target into a DataFrame
    data = pd.DataFrame({'Feature1': feature1, 'Feature2': feature2, 'Target': target})

    return data

def export_to_csv(data, file_path):
    data.to_csv(file_path, index=False)

# Generate the dataset with 1000 samples
dataset = generate_synthetic_dataset(num_samples=1000)

# Display the first few rows of the dataset
print(dataset.head())

# Export the dataset to a CSV file
file_path = 'filepath'  # Replace with the actual file path and name
export_to_csv(dataset, file_path)

print("Dataset exported to CSV successfully.")
