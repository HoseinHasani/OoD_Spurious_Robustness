import pandas as pd
import numpy as np
from tqdm import tqdm

threshold = 12

df = pd.read_csv('dataset_summary.csv', index_col=0)
# df = pd.read_csv('reduced_th_4.csv', index_col=0)
print(df.shape)

merged_row_attrs = np.load('merged_row_attrs.npy', allow_pickle=True).item()
merged_col_attrs = np.load('merged_col_attrs.npy', allow_pickle=True).item()


row_mapping = {label: merged_row_attrs.get(label, label) for label in df.index}
col_mapping = {label: merged_col_attrs.get(label, label) for label in df.columns}

df['merged_label_row'] = df.index.map(row_mapping)
df_grouped_rows = df.groupby('merged_label_row').sum()  

df_grouped_rows = df_grouped_rows.T
df_grouped_rows['merged_label_col'] = df_grouped_rows.index.map(col_mapping)
df_grouped = df_grouped_rows.groupby('merged_label_col').sum().T

# if 'merged_label' in df_grouped.columns:
#     df_grouped = df_grouped.drop(columns=['merged_label'])

data = df_grouped.values

row_names = np.array(df_grouped.index.tolist())
col_names = np.array(df_grouped.columns.tolist())

selected1 = []


print(df_grouped.shape)
binary_matrix = (data >= threshold).astype(int)

n_iter_r0 = 1288

# row_similarity_matrix = np.dot(binary_matrix, binary_matrix.T)
# row_similarity_array = row_similarity_matrix.sum(-1)
row_density_array = binary_matrix.sum(-1)
row_pop_array = data.sum(-1)
row_score = 0.2 * row_density_array / np.sum(row_density_array)\
        + 0.8 * row_pop_array / np.sum(row_pop_array) \
        # + 0.1 * row_similarity_array / np.sum(row_similarity_array)
        
arg_mins = np.argsort(row_score)[:n_iter_r0]
inds = np.arange(len(row_score))
new_inds = np.delete(inds, arg_mins)

binary_matrix = binary_matrix[new_inds]
data = data[new_inds]
row_names = row_names[new_inds]

n_iter_c = 470

for i in tqdm(range(n_iter_c)):
    
    n_iter_r = 2
    
    for j in range(n_iter_r):
        # row_similarity_matrix = np.dot(binary_matrix, binary_matrix.T)
        # row_similarity_array = row_similarity_matrix.sum(-1)
        row_density_array = binary_matrix.sum(-1)
        row_pop_array = data.sum(-1)
        row_score = 0.2 * row_density_array / np.sum(row_density_array)\
                + 0.8 * row_pop_array / np.sum(row_pop_array) \
                # + 0.1 * row_similarity_array / np.sum(row_similarity_array)
                
        arg_min = np.argwhere(row_score == np.min(row_score)).ravel()[-1]
        inds = np.arange(len(row_score))
        new_inds = np.delete(inds, arg_min)
        
        binary_matrix = binary_matrix[new_inds]
        data = data[new_inds]
        row_names = row_names[new_inds]
        
    
    # col_similarity_matrix = np.dot(binary_matrix.T, binary_matrix)
    # col_similarity_array = col_similarity_matrix.sum(0)
    col_density_array = binary_matrix.sum(0)
    col_pop_array = data.sum(0)
    col_score = 0.1 * col_density_array / np.sum(col_density_array)\
            + 0.9 * col_pop_array / np.sum(col_pop_array)\
            # + 0.05 * col_similarity_array / np.sum(col_similarity_array)   
            
    arg_min = np.argwhere(col_score == np.min(col_score)).ravel()[-1]
    inds = np.arange(len(col_score))
    new_inds = np.delete(inds, arg_min)
    
    binary_matrix = binary_matrix[:, new_inds]
    data = data[:, new_inds]
    col_names = col_names[new_inds]

binary_df = pd.DataFrame(data, index=row_names, columns=col_names)

binary_df.to_csv(f'reduced_th_{threshold}.csv')
