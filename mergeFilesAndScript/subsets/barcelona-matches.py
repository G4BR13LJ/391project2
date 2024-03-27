import pandas as pd

# Path to the CSV file
csv_file_path = '../master_dataset.csv'

# Import CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Filter where Barcelona is either the home team or away team
barcelona_matches = df.loc[(df['HomeTeam'] == 'Barcelona') | (df['AwayTeam'] == 'Barcelona')]

# Export DataFrame to CSV file
barcelona_matches.to_csv('barcelona-matches.csv', index=False)
