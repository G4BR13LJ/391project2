import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

# Read the dataset
df = pd.read_csv('../master_dataset.csv')

# Extract relevant columns for analysis
columns_for_analysis = ['HomeTeam', 'AwayTeam', 'FTR']

# Frequent pattern analysis
data_for_analysis = df[columns_for_analysis].astype(str)

# Convert categorical data into binary format (one-hot encoding)
te = TransactionEncoder()
data_encoded = te.fit_transform(data_for_analysis.apply(lambda x: x.dropna().tolist(), axis=1))
df_encoded = pd.DataFrame(data_encoded, columns=te.columns_)

# Apply Apriori algorithm to find frequent itemsets
min_support = 0.2
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

# Decode the frequent itemsets
decoded_itemsets = []
for itemset in frequent_itemsets['itemsets']:
    decoded_itemset = set()
    for item in itemset:
        original_columns = [col for col in te.columns_ if col.startswith(item)]
        decoded_itemset.update(original_columns)
    decoded_itemsets.append(decoded_itemset)

# Visualization of frequent itemsets
plt.figure(figsize=(20, 14))
plt.barh(range(len(frequent_itemsets)), frequent_itemsets['support'], align='center')
plt.yticks(range(len(frequent_itemsets)), [', '.join(itemset) for itemset in decoded_itemsets])
plt.xlabel('Support')
plt.title(f'Frequent Itemsets (Support > {min_support})')
plt.show()
