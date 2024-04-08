import pandas as pd

# Read the dataset
df = pd.read_csv('../master_dataset.csv')

# Calculate the average number of corners
average_corners = (df['HC'].sum() + df['AC'].sum()) / len(df)

print("Average Number of Corner Kicks (AC + HC):", average_corners)