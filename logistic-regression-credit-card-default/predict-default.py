import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE

# -- Data Import -- #

# import training data
training_data = pd.read_csv('/Users/loughlinrodd/Desktop/ai-problems/predict-credit-card-default/default_of_credit_card_clients.csv')

# print the first few rows of training_data 
print(training_data.head())

# -- Calculate baseline metrics for default rates -- #

# Default rate of women #
women = training_data.loc[training_data.SEX == 2]["default payment next month"]
rate_women = sum(women)/len(women)
print("% of women who default: ", rate_women)
# Default rate of men #
men = training_data.loc[training_data.SEX == 1]["default payment next month"]
rate_men = sum(men)/len(men)
print("% of men who default: ", rate_men)

# -- Data Preprocessing -- #

# extract the target variable 'default payment next month' from training_data into y
y = training_data['default payment next month']
# copy training_data into X without the target variable column
X = training_data.drop('default payment next month', axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data so that all feature variables have the same scale
# initialize the Standard Scaler
scaler = StandardScaler()
# Use fit_transform to fit the data (computer the mean stdev of each feature variable)
# and transform it to standaradize the features using the mean and stdev
X_train = scaler.fit_transform(X_train)
# call transform on the test data - do not call fit so that the mean and stdev 
# computed from the training data are used
X_test = scaler.transform(X_test)



# -- Build the logistic regression model -- ##
# Default max_iter is 100, but we have a complex data set, so we use 1000
logreg = LogisticRegression(max_iter=1000) # Increase max_iter for better convergence

# Initialize recursive feature selection with the logistic regression model
# Here, we aim to select the top 10 features. You can adjust this number accordingly.
selector = RFE(estimator=logreg, n_features_to_select=15, step=1)

# Fit RFE
selector = selector.fit(X_train, y_train)

# Transform training and testing sets so that they only include the selected features
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# train the logistic regression model on training data 
# Fit() djusts the model's weights using the training data to minimize prediction error
logreg.fit(X_train_selected, y_train)

# Once the model is trained, predict outcomes for the test data and store in y_pred
y_pred = logreg.predict(X_test_selected)


# -- Evaluate model accuracy -- #
# accurracy_score calculates the accuracy of predictions by comparing
# predicted values from y_pred to the true values in y_test and returns
# the correctly predcited samples
accuracy = accuracy_score(y_test, y_pred)
# print the accuracry score
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# print the ranking of features. The ones ranked 1 are selected.
print("Feature Ranking:", selector.ranking_)
