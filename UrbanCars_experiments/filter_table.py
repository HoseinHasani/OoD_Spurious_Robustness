import pandas as pd
import numpy as np

threshold = 8

df = pd.read_csv('dataset_summary.csv', index_col=0)
merged_row_attrs = np.load('merged_row_attrs.npy', allow_pickle=True).item()


row_mapping = {label: merged_row_attrs.get(label, label) for label in df.index}

df['merged_label'] = df.index.map(row_mapping)
df_grouped = df.groupby('merged_label').sum()  # Use .mean() if averaging is preferred

if 'merged_label' in df_grouped.columns:
    df_grouped = df_grouped.drop(columns=['merged_label'])

data = df_grouped.values


row_names = np.array(df_grouped.index.tolist())
col_names = np.array(df_grouped.columns.tolist())

binary_matrix = (data >= threshold).astype(int)


for i in range(48):
    
    n_iter = 105 if i == 0 else 10
    
    for j in range(n_iter):
        row_similarity_matrix = np.dot(binary_matrix, binary_matrix.T)
        row_similarity_array = row_similarity_matrix.sum(-1)
        row_density_array = binary_matrix.sum(-1)
        row_score = row_density_array / np.sum(row_density_array) + row_similarity_array / np.sum(row_similarity_array)
        arg_min = np.argwhere(row_score == np.min(row_score)).ravel()[-1]
        inds = np.arange(len(row_score))
        new_inds = np.delete(inds, arg_min)
        
        binary_matrix = binary_matrix[new_inds]
        data = data[new_inds]
        row_names = row_names[new_inds]
        
    
    col_similarity_matrix = np.dot(binary_matrix.T, binary_matrix)
    col_similarity_array = col_similarity_matrix.sum(-1)
    col_density_array = binary_matrix.sum(0)
    col_score = col_density_array / np.sum(col_density_array) + col_similarity_array / np.sum(col_similarity_array)
    arg_min = np.argwhere(col_score == np.min(col_score)).ravel()[-1]
    inds = np.arange(len(col_score))
    new_inds = np.delete(inds, arg_min)
    
    binary_matrix = binary_matrix[:, new_inds]
    data = data[:, new_inds]
    col_names = col_names[new_inds]
        
binary_df = pd.DataFrame(data, index=row_names, columns=col_names)

binary_df.to_csv(f'reduced_th_{threshold}.csv')
