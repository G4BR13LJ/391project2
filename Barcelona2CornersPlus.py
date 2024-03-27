import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
barcelona_data = pd.read_csv("desktop/barcelona-matches-full-columns.csv")

# Filter matches where Barcelona wins
barcelona_home_wins = barcelona_data[(barcelona_data["Home Team"] == "Barcelona") & (barcelona_data["Full Time Result (H=Home Win, D=Draw, A=Away Win)"] == "H")]
barcelona_away_wins = barcelona_data[(barcelona_data["Away Team"] == "Barcelona") & (barcelona_data["Full Time Result (H=Home Win, D=Draw, A=Away Win)"] == "A")]

# Concatenate both dataframes
barcelona_wins = pd.concat([barcelona_home_wins, barcelona_away_wins])

# Filter matches where Barcelona takes at least 2 corner shots
barcelona_wins_with_corners = barcelona_wins[barcelona_wins["Home Team Corners"] >= 2]

# Select relevant features and target variable
features = ["Home Team Corners"]
target = "Full Time Result (H=Home Win, D=Draw, A=Away Win)"

# Remove rows with missing values
barcelona_wins_with_corners = barcelona_wins_with_corners.dropna(subset=[*features, target])

# Convert target variable to binary outcome (1 for home win, 0 otherwise)
barcelona_wins_with_corners[target] = (barcelona_wins_with_corners[target] == "H").astype(int)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(barcelona_wins_with_corners[features], barcelona_wins_with_corners[target], test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
