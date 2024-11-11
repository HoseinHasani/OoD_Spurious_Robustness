import numpy as np


attrs = ['at home', 'autumn', 'dim', 'grass', 'in cage', 'on snow', 'rock', 'water']

n_train_dict = {
    'cat': [],
    'cow': [],
    'crab': [],
    'dog': [],
    'dolphin': [],
    'elephant': [],
    'fox': [],
    'frog': [],
    'giraffe': [],
    'goose': [],
    'horse': [],
    'lion': [],
    'monkey': [],
    'owl': [],
    'rabit': [],
    'rat': [],
    'seal': [],
    'sheep': [],
    'squirrel': [],
    'tiger': [],
    'wolf': [],
    }

np.save('class_train_num', n_train_dict)