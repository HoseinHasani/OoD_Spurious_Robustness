import os
import numpy as np
import pandas as pd
import random

n_train_dict = np.load('class_train_num.npy', allow_pickle=True).item()

metadata = pd.read_csv('dataset_metadata.csv')

updated_metadata = []

classes = metadata['class'].unique()
# attributes = list(metadata['attribute'].unique())
attributes = ['at home', 'autumn', 'dim', 'grass', 'in cage', 'on snow', 'rock', 'water']

for class_name in classes:
    for attribute_name in attributes:
        group = metadata[(metadata['class'] == class_name) & (metadata['attribute'] == attribute_name)]
        
        attribute_index = attributes.index(attribute_name)
        # n_train = n_train_dict.get(class_name, [0] * len(attributes))[attribute_index]
        n_train = n_train_dict[class_name][attribute_index]
        group_indices = group.index.tolist()
        random.shuffle(group_indices)
        
        train_indices = group_indices[:n_train]
        remaining_indices = group_indices[n_train:]
        n_remaining = len(remaining_indices)

        if n_remaining < 10:
            test_indices = remaining_indices  # All remaining to test
            val_indices = []
        elif n_remaining < 20:
            n_test = int(0.8 * n_remaining)
            test_indices = random.sample(remaining_indices, n_test)
            val_indices = list(set(remaining_indices) - set(test_indices))
        elif n_remaining < 30:
            n_test = int(0.6 * n_remaining)
            test_indices = random.sample(remaining_indices, n_test)
            val_indices = list(set(remaining_indices) - set(test_indices))
        else:
            n_test = min(int(0.5 * n_remaining), 25)
            test_indices = random.sample(remaining_indices, n_test)
            val_indices = list(set(remaining_indices) - set(test_indices))
        
        metadata.loc[train_indices, 'split'] = 0  
        metadata.loc[val_indices, 'split'] = 1     
        metadata.loc[test_indices, 'split'] = 2    

metadata.to_csv('updated_dataset_metadata.csv', index=False)
print("Updated metadata saved to 'updated_dataset_metadata.csv'")
