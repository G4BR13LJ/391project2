import pandas as pd

# Path to the CSV file
csv_file_path = './master_dataset.csv'

# Import CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)


##### Home versus away win/loss/tie percentages #####

# Filter rows where home wins, away wins, draws
home_wins = df[df['FTR'] == "H"]
away_wins = df[df['FTR'] == "A"]
ties = df[df['FTR'] == "D"]

# Calculate win/tie percentages
home_win_percentage = len(home_wins) / len(df) * 100
away_win_percentage = len(away_wins) / len(df) * 100
tie_percentage = len(ties) / len(df) * 100

# Print percentages
print(f"Home team win percentage: {home_win_percentage:.2f}%")
print(f"Away team win percentage: {away_win_percentage:.2f}%")
print(f"Draws percentage: {tie_percentage:.2f}%")


##### Results after leading at halftime #####

# Filter rows of teams leading or tied at half
home_leading_at_half = df[df['HTR'] == "H"]
away_leading_at_half = df[df['HTR'] == "A"]
#tied_at_half = df[df['HTR'] == "D"]

# Get all matches where team won after leading at half
home_wins_leading_at_half = home_leading_at_half[home_leading_at_half['FTR'] == "H"]
away_wins_leading_at_half = away_leading_at_half[away_leading_at_half['FTR'] == "A"]

# Get matches where a draw happened
home_ties_after_leading_at_half = home_leading_at_half[home_leading_at_half['FTR'] == "D"]
away_ties_after_leading_at_half = away_leading_at_half[away_leading_at_half['FTR'] == "D"]

# Win percentages after leading at half
home_wins_percentage_leading_at_half = len(home_wins_leading_at_half) / len(home_leading_at_half) * 100
away_wins_percentage_leading_at_half = len(away_wins_leading_at_half) / len(away_leading_at_half) * 100
home_ties_percentage_leading_at_half = len(home_ties_after_leading_at_half) / len(home_leading_at_half) * 100
away_ties_percentage_leading_at_half = len(away_ties_after_leading_at_half) / len(away_leading_at_half) * 100

# Print percentages
print(f"Home wins percentage after leading at half: {home_wins_percentage_leading_at_half:.2f}%")
print(f"Away wins percentage after leading at half: {away_wins_percentage_leading_at_half:.2f}%")
print(f"Draws percentage after home team leading at half: {home_ties_percentage_leading_at_half:.2f}%")
print(f"Draws percentage after away team leading at half: {away_ties_percentage_leading_at_half:.2f}%")