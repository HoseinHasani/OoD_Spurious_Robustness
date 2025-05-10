import pickle
import os

path = 'pickles'  

for filename in os.listdir(path):
    if filename.endswith('.pkl'):
        # if 's14' not in filename:
        #     continue
        if 'clbood' not in filename:
            continue
        
        file_path = os.path.join(path, filename)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(f'{filename}:')
            print(data['AUROC'])
            print('-' * 40)
