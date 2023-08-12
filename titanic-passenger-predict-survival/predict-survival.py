import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -- Data Import -- #

# import training data
training_data = pd.read_csv('/Users/loughlinrodd/Desktop/ai-problems/titanic-passenger-predict-survival/train.csv')

# print the first few rows of training_data 
print(training_data.head())

# import testing data
testing_data = pd.read_csv('/Users/loughlinrodd/Desktop/ai-problems/titanic-passenger-predict-survival/test.csv')

# print the first few rows of testing_data
print(testing_data.head())

# -- Calculate baseline survival rates of women and men -- #

# determine percentage of women that survived and print
# select column Sex and filter on value of fameale and select the survived column for these rows
women = training_data.loc[training_data.Sex == 'female']["Survived"]
# sum the women column from the series to get the number of women who survived and get the length of the column to get the number of women on the Titanic
# divide the sum by the length to get the surival rate
rate_women = sum(women)/len(women)

print("% of women who survived: ", rate_women)

# determine percentage of men that survived and print
men = training_data.loc[training_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived: ", rate_men)

# -- Data Preprocessing -- #

# extract the target variable Survived from training_data into y
y = training_data["Survived"]

# select feature variables into list 
features = ["Pclass", "Sex", "SibSp", "Parch"]

# use pandas get_dummies() to convert categorical variables from features into indicator variables 
# using one-hot encoding, one-hot encoding will taxe a feature variables like Sex and convert it into 
# two seperate columns, Sex_male and Sex_female with binary values
# X containes one-hot encoded feature variables of training data
X = pd.get_dummies(training_data[features])
# X_test containes one-hot encoded feature variables of testing data
X_test = pd.get_dummies(testing_data[features])


model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X,y) 
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': testing_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Submission saved successfully :)")
