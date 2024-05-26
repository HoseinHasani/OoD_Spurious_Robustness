import numpy as np
from sklearn.metrics import accuracy_score

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

test_dict = np.load('x_test_group_dict.npy', allow_pickle=True).item()

prototypes = []
for l in [0, 1]:
    inds = np.argwhere(y_train == l).ravel()
    prototypes.append(x_train[inds].mean(0))

prototypes = np.array(prototypes).T

for key in test_dict.keys():
    data = test_dict[key]
    label = float(key[:1]) * np.ones(len(data))
    
    dists = np.linalg.norm(data[..., None] - prototypes[None], axis=1)
    preds = np.argmin(dists, -1)    
    
    acc = np.round(accuracy_score(label, preds), 5)
    print(f'group: {key}, acc: {acc}')