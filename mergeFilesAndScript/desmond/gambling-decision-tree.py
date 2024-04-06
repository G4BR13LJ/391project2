#!/usr/bin/env python
"""
This script creates a decision tree for determining the result
of a football match based on betting odds.

Author: Desmond Stular
  Date: April 6, 2024
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import graphviz

# Load the dataset
df = pd.read_csv('../master_dataset.csv')

features = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA']


# Select features and target variable
x = df[features]
y = df['FTR']       # Full Time Result

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.258, random_state=9)

# Create and train the decision tree model
clf = DecisionTreeClassifier(random_state=9)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Decision tree prediction accuracy: {accuracy}")

# Get the predicted probabilities for the test set
y_pred_prob = clf.predict_proba(X_test)

# Get the maximum predicted probability for each sample
max_pred_prob = y_pred_prob.max(axis=1)

# Calculate the average maximum predicted probability
average_max_prob = max_pred_prob.mean()
print(f"Average maximum predicted probability: {average_max_prob}")

### Using graphviz to draw the decision tree ###

# feature_names = X_train.columns.tolist()
#
# class_names = clf.classes_
#
# export_graphviz(clf, out_file="tree.dot",
#                 feature_names=feature_names, # Replace with your actual feature names
#                 class_names=class_names,      # Replace with your actual class names
#                 filled=True, rounded=True,
#                 rankdir="TB")
#
#
# # Display the decision tree
# graphviz.Source.from_file("tree.dot").view("gambling tree")