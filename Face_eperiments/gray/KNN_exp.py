import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import dist_utils
import os
import warnings
import pickle

warnings.filterwarnings("ignore")

normalize_embs = True

k_values = [0, 1, 7, 10]



core_class_names = ['0', '1']
ood_class_names = ['0', '1']
sp_class_names = ['0', '1']

backbones = ['clip_ViT_B_16', 'dinov2_vits14', 'clip_ViT_B_16']

backbone = backbones[0]

in_data_embs0 = np.load(f'embs/celeba_embs_{backbone}_0.8_seed14.npy', allow_pickle=True).item()
    
ood_embs0 = np.load(f'embs/ood_embeddings_{backbone}.npy', allow_pickle=True).item()['clbood']

ood_embs_list = []
for k in ood_embs0.keys():
    ood_embs_list.append(ood_embs0[k])
    
ood_embs0 = {}

ood_embs0['0'] = np.array(ood_embs_list[:len(ood_embs_list) //2])
ood_embs0['1'] = np.array(ood_embs_list[len(ood_embs_list) //2:])

grouped_embs0 = {}
grouped_embs_train0 = {}
grouped_embs_val0 = {}

for key in in_data_embs0.keys():
    emb = in_data_embs0[key].squeeze()
    label = key[0]
    place = int(key[2])
    if place > 1:
        place = place - 2
    split = key[4]
    name = f'{label}_{place}'
    
    if split == '2':
        if name not in grouped_embs0.keys():
            grouped_embs0[name] = []
        
        grouped_embs0[name].append(emb)
    elif split == '1':
        if name not in grouped_embs_val0.keys():
            grouped_embs_val0[name] = []
        
        grouped_embs_val0[name].append(emb)
    else:
        if name not in grouped_embs_train0.keys():
            grouped_embs_train0[name] = []
        
        grouped_embs_train0[name].append(emb)

grouped_embs0 = {name: np.array(grouped_embs0[name]) for name in grouped_embs0.keys()}
grouped_embs_train0 = {name: np.array(grouped_embs_train0[name]) for name in grouped_embs_train0.keys()}
grouped_embs_val0 = {name: np.array(grouped_embs_val0[name]) for name in grouped_embs_val0.keys()}


grouped_embs0 = {name: np.array(grouped_embs0[name]) for name in grouped_embs0.keys()}
grouped_embs_train0 = {name: np.array(grouped_embs_train0[name]) for name in grouped_embs_train0.keys()}
grouped_embs_val0 = {name: np.array(grouped_embs_val0[name]) for name in grouped_embs_val0.keys()}



def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


grouped_embs = {name: grouped_embs0[name] for name in grouped_embs0.keys()}
grouped_embs_train0 = grouped_embs.copy()
ood_embs = {name: ood_embs0[name] for name in ood_embs0.keys()}


if normalize_embs:
    train_dict = {key: normalize(grouped_embs_train0[key]) for key in grouped_embs_train0.keys()}
    test_dict = {key: normalize(grouped_embs0[key]) for key in grouped_embs0.keys()}
    ood_dict = {key: normalize(ood_embs0[key]) for key in ood_embs0.keys()}
else:
    train_dict = grouped_embs_train0
    test_dict = grouped_embs
    ood_dict = ood_embs0





def get_class_dicts(input_dict):
    class_dicts = []
    for core_name in core_class_names:
        class_dict = {}
        for sp_name in sp_class_names:
            name = f'{core_name}_{sp_name}'
            class_dict[name] = input_dict[name]
        class_dicts.append(class_dict)

    return class_dicts


def plot_dict_hist(dict_data, fig_name):
    data = []
    for name in dict_data:
        data.append(dict_data[name])
    
    data = np.concatenate(data)
    
    plt.hist(data, 100, histtype='step', linewidth=1.5, label=fig_name)
    plt.legend()

def get_nn_distances(targets, sources):
    dists = []
    for target in targets:
        dists_ = [np.linalg.norm(source - target) for source in sources]
        dists.append(np.sort(dists_)[:100])
        
    return np.array(dists)
    
def knn_classifier(dists, k=5):
    preds = []
    th_dists = []
    for i in range(dists.shape[1]):
        dists_ = dists[:, i]
        th_val = np.sort(dists_.ravel())[k]
        th_dists.append(th_val)
        n_cl0 = len(np.argwhere(dists_[0] <= th_val).ravel())
        n_cl1 = len(np.argwhere(dists_[1] <= th_val).ravel())
        preds.append(int(n_cl1 > n_cl0))
    
    return np.array(preds), np.array(th_dists)

    
train_dict_list = get_class_dicts(train_dict)
test_dict_list = get_class_dicts(test_dict)

print('K-NN accuracy:')

test_knn_dists = {k: [] for k in k_values}
for key in test_dict.keys():
    
    targets = test_dict[key]
    labels = float(key[0]) * np.ones(len(test_dict[key]))
    
    dists = []
    for c in range(len(train_dict_list)):
        sources = np.concatenate([train_dict_list[c][kk] for kk in train_dict_list[c].keys()])
        dists.append(get_nn_distances(targets, sources))
    dists = np.array(dists)
    knn_accs = []
    for k in k_values:
        preds, knn_dists_ = knn_classifier(dists, k)
        test_knn_dists[k].append(knn_dists_)
        knn_accs.append(accuracy_score(labels, preds))
    
    print(key, np.round(knn_accs, 5))

test_knn_dists = {k: np.concatenate(test_knn_dists[k]) for k in k_values}

ood_embs = np.concatenate([ood_dict[key] for key in ood_dict.keys()])


ood_knn_dists = {k: [] for k in k_values}

dists = []
for c in range(len(train_dict_list)):
    sources = np.concatenate([train_dict_list[c][kk] for kk in train_dict_list[c].keys()])
    dists.append(get_nn_distances(ood_embs, sources))
dists = np.array(dists)
for k in k_values:
    preds, knn_dists_ = knn_classifier(dists, k)
    ood_knn_dists[k].append(knn_dists_)

ood_knn_dists = {k: np.concatenate(ood_knn_dists[k]) for k in k_values}


print('OOD:')
for k in k_values:
    print(backbone, normalize_embs, k)
    dist_utils.calc_ROC_with_dists(test_knn_dists[k], ood_knn_dists[k])


