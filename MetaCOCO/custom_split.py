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





n_train_dict = np.load('class_train_num.npy', allow_pickle=True).item()
updated_metadata = pd.read_csv('updated_dataset_metadata.csv', index_col=0)
original_metadata = pd.read_csv('dataset_metadata.csv', index_col=0)

validation_passed = True

# 1. Check if number of training samples in each group matches n_train_dict
for class_name, train_counts in n_train_dict.items():
    for attribute_index, n_train in enumerate(train_counts):
        attribute_name = attributes[attribute_index]  # Get attribute name by index
        
        train_samples = updated_metadata[
            (updated_metadata['class'] == class_name) &
            (updated_metadata['attribute'] == attribute_name) &
            (updated_metadata['split'] == 0)
        ]
        
        if len(train_samples) != n_train:
            print(f"Mismatch in training count for class '{class_name}', attribute '{attribute_name}':"
                  f" Expected {n_train}, found {len(train_samples)}.")
            validation_passed = False

# 2. Check if indices, address, class, and attribute match between updated and original metadata
for column in ['address', 'class', 'attribute']:
    if not updated_metadata[column].equals(original_metadata[column]):
        print(f"Mismatch detected in column '{column}': original and updated metadata are not in the same order or contain mismatched values.")
        validation_passed = False

if validation_passed:
    print("All validations passed: Training sample counts match, and data integrity is preserved in the updated metadata.")
else:
    print("Some validations failed. Please check the log above for mismatches.")
    
    
    
