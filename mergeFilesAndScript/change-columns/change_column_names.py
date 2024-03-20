import pandas as pd
import column_mapping

# Path to the CSV file
csv_file_path = '../master_dataset.csv'

# Import CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Rename columns using the dictionary
df_renamed = df.rename(columns=column_mapping.column_dict)

# Path to save the CSV file
csv_file_path = './master_full_columns.csv'

# Export the DataFrame to a CSV file
df_renamed.to_csv(csv_file_path, index=False)