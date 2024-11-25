import os
import pickle
import numpy as np


path = 'results/'
files = os.listdir(path)
for file_name in files:
    if 'summary' not in file_name:
        continue

    if 'animal' not in file_name:
        continue
    
    if file_name[-4:] != '.pkl':
        continue
    with open(f'{path}/{file_name}', 'rb') as handle:
        res_dict = pickle.load(handle)
    print('********************')
    print(file_name)
    try:
        keys = list(res_dict.keys())
        # ind = np.random.choice(len(keys), 1).item()
        for key in keys:
            print(key)
            print(res_dict[key])
        
            
    except:
        continue