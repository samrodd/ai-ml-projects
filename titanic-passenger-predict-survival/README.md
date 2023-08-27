Code uses a randomForestClassifier to predict the survival of passengers on the Titanic.
The predictions are from the classifier are output into submission.csv
test.csv is the testing data for the classifier, it does not have a column indicating whether the passenger survived.
traain.csv is the training data for the classifier, it does have a column indicating whether the passenger survived.

A randomForestClassifier builds multiple decision trees from which is aggregates predictions. Each tree is constructed with a different subset
of the training data. The classifier uses feature randomness by only using a random subset of the feature data on each tree. This helps ensure
the model is not overfitted. randomForestClassifiers can either perform classification or regression tasks. Here we ware classifying
passengers into survived and did not survive - in this case, the classifier makes its final decision by majority vote of the different trees
in the forest. 
