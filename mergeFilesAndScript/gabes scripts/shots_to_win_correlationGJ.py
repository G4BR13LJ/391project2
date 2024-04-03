import pandas as pd

# Read the dataset
df = pd.read_csv('../master_dataset.csv')

# Calculate the total number of occurrences for each value in the 'HC' column
hc_counts = df['HS'].value_counts()

# Create a DataFrame to store the probabilities
prob_df = pd.DataFrame(columns=['HS', 'Probability of H in FTR'])

# Iterate over unique values in the 'HC' column
for hc_value in sorted(df['HS'].unique()):
    # Filter the DataFrame based on the 'HC' value
    filtered_df = df[df['HS'] == hc_value]

    # Count the occurrences of 'H' in the 'FTR' column
    h_count = filtered_df[filtered_df['FTR'] == 'H'].shape[0]

    # Calculate the probability of 'H' in the 'FTR' column
    if hc_value in hc_counts:
        probability = h_count / hc_counts[hc_value]
    else:
        probability = 0.0

    # Append the result to the DataFrame
    prob_df.loc[len(prob_df)] = [hc_value, probability]

# Print the result
print("Probability of 'H' in FTR based on HS value:")
print(prob_df)

# ---------------------------------------------------------------------------------
# Calculate the total number of occurrences for each value in the 'HC' column
ac_counts = df['AS'].value_counts()

# Create a DataFrame to store the probabilities
prob1_df = pd.DataFrame(columns=['AS', 'Probability of A in FTR'])

# Iterate over unique values in the 'AC' column
for ac_value in sorted(df['AS'].unique()):
    # Filter the DataFrame based on the 'HC' value
    filtered1_df = df[df['AS'] == ac_value]

    # Count the occurrences of 'H' in the 'FTR' column
    a_count = filtered1_df[filtered1_df['FTR'] == 'A'].shape[0]

    # Calculate the probability of 'H' in the 'FTR' column
    if ac_value in ac_counts:
        probability = a_count / ac_counts[ac_value]
    else:
        probability = 0.0

    # Append the result to the DataFrame
    prob1_df.loc[len(prob1_df)] = [ac_value, probability]

# Print the result
print("Probability of 'A' in FTR based on AS value:")
print(prob1_df)
