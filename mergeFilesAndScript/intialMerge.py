import pandas as pd

# Read all 10 datasets into separate DataFrames
sp1df = pd.read_csv('SP1.csv')
sp2df = pd.read_csv('SP2.csv')
sp3df = pd.read_csv('SP3.csv')
sp4df = pd.read_csv('SP4.csv')
sp5df = pd.read_csv('SP5.csv')
sp6df = pd.read_csv('SP6.csv')
sp7df = pd.read_csv('SP7.csv')
sp8df = pd.read_csv('SP8.csv')
sp9df = pd.read_csv('SP9.csv')
sp10df = pd.read_csv('SP10.csv')

# Identify common columns among all datasets
common_columns = set(sp1df.columns)
for df in [sp2df, sp3df, sp4df, sp5df, sp6df, sp7df, sp8df, sp9df, sp10df]:
    common_columns = common_columns.intersection(df.columns)

# Create a list of common column names
common_columns = list(common_columns)

# Merge datasets using common columns
master_df = pd.concat([sp1df[common_columns], sp2df[common_columns], sp3df[common_columns],
                       sp4df[common_columns], sp5df[common_columns], sp6df[common_columns],
                       sp7df[common_columns], sp8df[common_columns], sp9df[common_columns],
                       sp10df[common_columns]], ignore_index=True)

print("Columns in master:", master_df.columns)

# Print the number of columns in the master dataset
print("Number of columns in master dataset:", len(master_df.columns))

# Print the number of rows (entries) in the master dataset
print("Number of rows in master dataset:", len(master_df))

# Print the name of each column, the number of unique values, and the names of all unique values
for column in master_df.columns:
    unique_values = master_df[column].unique()
    print(f"Column: {column}")
    print(f"Number of unique values: {len(unique_values)}")
    print("Unique values:", unique_values)
    print()

# # Export master DataFrame to a CSV file
# master_df.to_csv('master_dataset.csv', index=False)

