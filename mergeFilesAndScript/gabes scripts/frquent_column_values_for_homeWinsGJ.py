import pandas as pd

# Load your dataset into a DataFrame (replace 'your_dataset.csv' with the actual path to your dataset)
df = pd.read_csv('../master_dataset.csv')

column_dict = {
    'Div': 'League Division',
    'Date': 'Match Date (dd/mm/yy)',
    'Time': 'Time of match kick off',
    'HomeTeam': 'Home Team',
    'AwayTeam': 'Away Team',
    'FTHG': 'Full Time Home Team Goals',
    'FTAG': 'Full Time Away Team Goals',
    'FTR': 'Full Time Result (H=Home Win, D=Draw, A=Away Win)',
    'HTHG': 'Half Time Home Team Goals',
    'HTAG': 'Half Time Away Team Goals',
    'HTR': 'Half Time Result (H=Home Win, D=Draw, A=Away Win)',
    'HS': 'Home Team Shots',
    'AS': 'Away Team Shots',
    'HST': 'Home Team Shots on Target',
    'AST': 'Away Team Shots on Target',
    'HF': 'Home Team Fouls Committed',
    'AF': 'Away Team Fouls Committed',
    'HC': 'Home Team Corners',
    'AC': 'Away Team Corners',
    'HY': 'Home Team Yellow Cards',
    'AY': 'Away Team Yellow Cards',
    'HR': 'Home Team Red Cards',
    'AR': 'Away Team Red Cards',
    'B365H': 'Bet365 home win odds',
    'B365D': 'Bet365 draw odds',
    'B365A': 'Bet365 away win odds',
    'BWH': 'Bet&Win home win odds',
    'BWD': 'Bet&Win draw odds',
    'BWA': 'Bet&Win away win odds',
    'IWH': 'Interwetten home win odds',
    'IWD': 'Interwetten draw odds',
    'IWA': 'Interwetten away win odds',
    'LBH': 'Ladbrokes home win odds',
    'LBD': 'Ladbrokes draw odds',
    'LBA': 'Ladbrokes away win odds',
    'PSH': 'Pinnacle home win odds',
    'PSD': 'Pinnacle draw odds',
    'PSA': 'Pinnacle away win odds',
    'WHH': 'William Hill home win odds',
    'WHD': 'William Hill draw odds',
    'WHA': 'William Hill away win odds',
    'VCH': 'VC Bet home win odds',
    'VCD': 'VC Bet draw odds',
    'VCA': 'VC Bet away win odds',
    'Bb1X2': 'Number of BetBrain bookmakers used',
    'BbMxH': 'Betbrain maximum home win odds',
    'BbAvH': 'Betbrain average home win odds',
    'BbMxD': 'Betbrain maximum draw odds',
    'BbAvD': 'Betbrain average draw win odds',
    'BbMxA': 'Betbrain maximum away win odds',
    'BbAvA': 'Betbrain average away win odds',
    'BbOU': 'Number of BetBrain bookmakers used o/u',
    'BbMx>2.5': 'Betbrain maximum over 2.5 goals',
    'BbAv>2.5': 'Betbrain average over 2.5 goals',
    'BbMx<2.5': 'Betbrain maximum under 2.5 goals',
    'BbAv<2.5': 'Betbrain average under 2.5 goals',
    'BbAH': 'Number of BetBrain bookmakers used AZN handi',
    'BbAHh': 'Betbrain size of handicap (home team)',
    'BbMxAHH': 'Betbrain maximum Asian handicap home team odds',
    'BbAvAHH': 'Betbrain average Asian handicap home team odds',
    'BbMxAHA': 'Betbrain maximum Asian handicap away team odds',
    'BbAvAHA': 'Betbrain average Asian handicap away team odds'
}

# Create an empty list to hold the results
results = []

# Define the value you want to analyze
value_to_analyze = 'H'

# Iterate over each column in the DataFrame
for column in df.columns:
    if column != 'FTR':  # Skip the target column itself
        # Filter the DataFrame to only include rows where FTR column matches the value to analyze
        filtered_df = df[df['FTR'] == value_to_analyze]

        # Count occurrences of each value in the filtered DataFrame
        value_counts = filtered_df[column].value_counts()

        # Get the most common value and its count
        most_common_value = value_counts.idxmax()
        most_common_count = value_counts.max()

        # Calculate the percentage of entries that have the most common value
        percentage = most_common_count / len(filtered_df) * 100

        # Append the results to the list
        results.append({'Column': column_dict[column],
                        'Most Common Value': most_common_value,
                        'Percentage': percentage})

# Convert the list of dictionaries into a DataFrame
results_df = pd.DataFrame(results)

# Display the results
for value in results:
    print(value)

# RUN ASSOCIATION RULES

