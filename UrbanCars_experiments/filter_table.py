import pandas as pd
import numpy as np

threshold = 7  

df = pd.read_csv('dataset_summary.csv', index_col=0)
merged_attrs = np.load('merged_attrs.npy', allow_pickle=True).item()

data = df.values

row_names = np.array(df.index.tolist())
col_names = np.array(df.columns.tolist())

binary_matrix = (data >= threshold).astype(int)


for i in range(21):

    n_iter = 34 if i == 0 else 24
    
    for j in range(n_iter):
        row_similarity_matrix = np.dot(binary_matrix, binary_matrix.T)
        row_similarity_array = row_similarity_matrix.sum(-1)
        arg_min = np.argwhere(row_similarity_array == np.min(row_similarity_array)).ravel()[-1]
        inds = np.arange(len(row_similarity_array))
        new_inds = np.delete(inds, arg_min)
        
        binary_matrix = binary_matrix[new_inds]
        data = data[new_inds]
        row_names = row_names[new_inds]
        
    
    col_similarity_matrix = np.dot(binary_matrix.T, binary_matrix)
    col_similarity_array = col_similarity_matrix.sum(-1)
    arg_min = np.argwhere(col_similarity_array == np.min(col_similarity_array)).ravel()[-1]
    inds = np.arange(len(col_similarity_array))
    new_inds = np.delete(inds, arg_min)
    
    binary_matrix = binary_matrix[:, new_inds]
    data = data[:, new_inds]
    col_names = col_names[new_inds]
        
binary_df = pd.DataFrame(data, index=row_names, columns=col_names)

binary_df.to_csv(f'reduced_th_{threshold}.csv')
