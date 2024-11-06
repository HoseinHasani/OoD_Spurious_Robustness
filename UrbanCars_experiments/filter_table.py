import pandas as pd
from itertools import combinations
from tqdm import tqdm

threshold = 1
file_path = 'dataset_summary.csv'  
df = pd.read_csv(file_path)

df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

filtered_df = df.loc[:, (df >= threshold).any(axis=0)]
filtered_df = filtered_df.loc[(filtered_df >= threshold).any(axis=1), :]

best_rows = []
best_columns = []

for start_row in range(len(filtered_df)):
    for start_col in range(len(filtered_df.columns)):
        rows = [start_row]
        cols = [start_col]
        
        for end_row in range(start_row, len(filtered_df)):
            if (filtered_df.iloc[end_row, cols] >= threshold).all():
                rows.append(end_row)
            else:
                break
        
        for end_col in range(start_col, len(filtered_df.columns)):
            if (filtered_df.iloc[rows, end_col] >= threshold).all():
                cols.append(end_col)
            else:
                break

        if len(rows) * len(cols) > len(best_rows) * len(best_columns):
            best_rows = rows
            best_columns = cols

largest_sub_table = filtered_df.iloc[best_rows, best_columns]


largest_sub_table.to_csv(f'reduced_threshold_{threshold}.csv', index=False)

