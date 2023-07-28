import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

def univariate_feature_selection(data, k=4):
    X = data.drop(['lender', 'Approved'], axis=1)
    y = data['Approved']
    
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    
    # Get feature scores and sort them in descending order
    feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_})
    feature_scores = feature_scores.sort_values(by='Score', ascending=False)
    
    selected_features = feature_scores['Feature'][:k].tolist()
    selected_f_values = feature_scores['Score'][:k].tolist()
    
    return selected_features, selected_f_values

# Load the CSV file
file_path = '/Users/loughlinrodd/Desktop/ai-problems/univariate-feature-selection/ufs-dataset.csv'
data = pd.read_csv(file_path)

# Create separate DataFrames for each lender
lenders = data['lender'].unique()
selected_features_per_lender = {}

for lender in lenders:
    lender_data = data[data['lender'] == lender]
    selected_features, selected_f_values = univariate_feature_selection(lender_data, k=4)  # Change k=4 here
    selected_features_per_lender[lender] = (selected_features, selected_f_values)

# Output the ranked features and ANOVA F-values for each lender
for lender, (features, f_values) in selected_features_per_lender.items():
    ranked_features_with_f_values = [(feature, f_value) for feature, f_value in zip(features, f_values)]
    ranked_features_with_f_values = ", ".join([f"{feature} (F-value: {f_value:.2f})" for feature, f_value in ranked_features_with_f_values])
    print(f"Ranked features for {lender}: {ranked_features_with_f_values}")
