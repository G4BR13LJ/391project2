import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv('../master_dataset.csv')

# Select relevant columns for analysis
selected_columns = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF',
                    'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A', 'FTR']

# Filter the dataset
df_selected = df[selected_columns]

# Data Preprocessing
# Handle non-numeric values
df_selected.replace('-', np.nan, inplace=True)  # Replace dashes with NaN
df_selected.dropna(inplace=True)  # Drop rows with NaN values

# Clustering
# Select features for clustering
X = df_selected.drop(columns=['FTR'])

# Standardize the features
X = (X - X.mean()) / X.std()

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0)
df_selected['Cluster'] = kmeans.fit_predict(X)

# Visualize clustering results
sns.scatterplot(data=df_selected, x='HS', y='AS', hue='Cluster')
plt.title('Clustering of Matches Based on Home Shots and Away Shots')
plt.xlabel('Home Shots')
plt.ylabel('Away Shots')
plt.show()

# Classification
# Select features (X) and target (y)
X = df_selected[['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST',
                 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
                 'B365H', 'B365D', 'B365A']]
y = df_selected['FTR']  # Full Time Result

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Print Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print Classification Report
print("\nClassification Report:")
print("Precision: Ability of the classifier to not label a positive sample as negative.")
print("Recall: Ability of the classifier to find all positive samples.")
print("F1-score: Weighted average of precision and recall.")
print("Support: Number of actual occurrences of the class in the specified dataset.")
print()
print(classification_report(y_test, y_pred))

# Print Confusion Matrix
print("Confusion Matrix:")
print("The confusion matrix shows the number of correct and incorrect predictions made by the classifier.")
print("Each row represents the actual class, while each column represents the predicted class.")
print()
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Away Win (A)', 'Actual Draw (D)', 'Actual Home Win (H)'],
                              columns=['Predicted Away Win (A)', 'Predicted Draw (D)', 'Predicted Home Win (H)'])
# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Away Win (A)', 'Predicted Draw (D)', 'Predicted Home Win (H)'],
            yticklabels=['Actual Away Win (A)', 'Actual Draw (D)', 'Actual Home Win (H)'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# Display the importance of each feature in the classifier
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_classifier.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
