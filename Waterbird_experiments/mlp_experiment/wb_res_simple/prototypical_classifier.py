import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from numpy.linalg import norm


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
    
    dists = norm(data[..., None] - prototypes[None], axis=1)
    preds = np.argmin(dists, -1)    
    
    acc = np.round(accuracy_score(label, preds), 5)
    print(f'group: {key}, acc: {acc}')
    
    
if False:
    clf = LogisticRegression()
    
    clf.fit(x_train, y_train)
    y_hat_train = clf.predict(x_train)
else:
    dists = norm(x_train[..., None] - prototypes[None], axis=1)
    y_hat_train = np.argmin(dists, -1)  

total_miss_inds = np.argwhere(y_hat_train != y_train).ravel()
total_crr_inds = np.argwhere(y_hat_train == y_train).ravel()
print(np.intersect1d(total_crr_inds, total_miss_inds, assume_unique=True))

aug_prototypes = []

for l in [0, 1]:
    class_inds = np.argwhere(y_train == l).ravel()
    class_miss_inds = np.intersect1d(class_inds, total_miss_inds, assume_unique=True)
    aug_prototypes.append(x_train[class_miss_inds].mean(0))
    class_crr_inds = np.intersect1d(class_inds, total_crr_inds, assume_unique=True)
    aug_prototypes.append(x_train[class_crr_inds].mean(0))

aug_prototypes = np.array(aug_prototypes).T


for key in test_dict.keys():
    data = test_dict[key]
    label = float(key[:1]) * np.ones(len(data))
    
    dists = norm(data[..., None] - aug_prototypes[None], axis=1)
    preds = np.argmin(dists, -1)    
    
    preds[preds < 1.5] = 0
    preds[preds > 1] = 1
    
    acc = np.round(accuracy_score(label, preds), 5)
    print(f'group: {key}, acc: {acc}')


    
def normalize(x):
    return x / norm(x, axis=-1, keepdims=True)

cr_diff1 = aug_prototypes[:, 3] - aug_prototypes[:, 0]
cr_diff2 = aug_prototypes[:, 2] - aug_prototypes[:, 1]
cr_ax = normalize(cr_diff1 + cr_diff2)

sp_diff1 = aug_prototypes[:, 1] - aug_prototypes[:, 0]
sp_diff2 = aug_prototypes[:, 2] - aug_prototypes[:, 3]
sp_ax = normalize(sp_diff1 + sp_diff2)

def cosine(a, b):
    return np.dot(a, b)/(norm(a)*norm(b))

print()
print(cosine(cr_diff1, cr_diff2))
print(cosine(sp_diff1, sp_diff2))
print()
print(cosine(sp_diff2, cr_diff2))
print(cosine(sp_diff1, cr_diff2))
print(cosine(sp_diff2, cr_diff1))
print(cosine(sp_diff1, cr_diff1))
print()

def refine_embs(embs, cr_ax, sp_ax, alpha=.0, beta=0.99):
    
    refined = 0.99 * embs.copy()
    cr_coefs = np.dot(embs, cr_ax)
    refined += alpha * cr_coefs[:, None] * np.repeat(cr_ax[None], embs.shape[0], axis=0)
    sp_coefs = np.dot(refined, sp_ax)
    refined -= beta * sp_coefs[:, None] * np.repeat(sp_ax[None], embs.shape[0], axis=0)
    
    return refined

x_train_refined = refine_embs(x_train, cr_ax, sp_ax)

prototypes_refined = []
for l in [0, 1]:
    inds = np.argwhere(y_train == l).ravel()
    prototypes_refined.append(x_train_refined[inds].mean(0))

prototypes_refined = np.array(prototypes_refined).T

for key in test_dict.keys():
    data = refine_embs(test_dict[key], cr_ax, sp_ax)
    label = float(key[:1]) * np.ones(len(data))
    
    dists = norm(data[..., None] - prototypes_refined[None], axis=1)
    preds = np.argmin(dists, -1)    
    
    acc = np.round(accuracy_score(label, preds), 5)
    print(f'group: {key}, acc: {acc}')