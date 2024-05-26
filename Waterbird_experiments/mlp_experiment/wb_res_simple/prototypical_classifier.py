import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


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
    
    
if False:
    clf = LogisticRegression()
    
    clf.fit(x_train, y_train)
    y_hat_train = clf.predict(x_train)
else:
    dists = np.linalg.norm(x_train[..., None] - prototypes[None], axis=1)
    y_hat_train = np.argmin(dists, -1)  

total_miss_inds = np.argwhere(y_hat_train != y_train).ravel()
total_crr_inds = np.argwhere(y_hat_train == y_train).ravel()
print(np.intersect1d(total_crr_inds, total_miss_inds, assume_unique=True))

prototypes = []

for l in [0, 1]:
    class_inds = np.argwhere(y_train == l).ravel()
    class_miss_inds = np.intersect1d(class_inds, total_miss_inds, assume_unique=True)
    prototypes.append(x_train[class_miss_inds].mean(0))
    class_crr_inds = np.intersect1d(class_inds, total_crr_inds, assume_unique=True)
    prototypes.append(x_train[class_crr_inds].mean(0))

prototypes = np.array(prototypes).T


for key in test_dict.keys():
    data = test_dict[key]
    label = float(key[:1]) * np.ones(len(data))
    
    dists = np.linalg.norm(data[..., None] - prototypes[None], axis=1)
    preds = np.argmin(dists, -1)    
    
    preds[preds < 1.5] = 0
    preds[preds > 1] = 1
    
    acc = np.round(accuracy_score(label, preds), 5)
    print(f'group: {key}, acc: {acc}')
