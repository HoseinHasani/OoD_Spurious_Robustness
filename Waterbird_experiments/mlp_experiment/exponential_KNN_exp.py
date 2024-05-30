import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import dist_utils
import os
import warnings

warnings.filterwarnings("ignore")

normalize_embs = True

alpha_values = [0.003, 0.01, 0.06, 0.12]

backbones = ['dino', 'res50', 'res18']
backbone = backbones[2]
resnet_types = ['pretrained', 'finetuned', 'scratch']
resnet_type = resnet_types[0]

core_class_names = ['0', '1']
ood_class_names = ['0', '1']
sp_class_names = ['0', '1']
place_names = ['land', 'water']

data_path = '../embeddings/'



if backbone == 'dino':
    in_data_embs0 = np.load(data_path + 'waterbird_embs_DINO.npy', allow_pickle=True).item()
else:
    in_data_embs0 = np.load(data_path + f'wb_embs_{backbone}_{resnet_type}.npy', allow_pickle=True).item()

ood_embs0 = {}
if backbone == 'dino':
    dict_ = np.load(data_path + 'OOD_land_DINO_eval.npy', allow_pickle=True).item()
else:
    dict_ = np.load(data_path + f'OOD_land_{backbone}_eval.npy', allow_pickle=True).item()
    
ood_embs0['0'] = np.array([dict_[key].squeeze() for key in dict_.keys()])

if backbone == 'dino':
    dict_ = np.load(data_path + 'OOD_water_DINO_eval.npy', allow_pickle=True).item()
else:
    dict_ = np.load(data_path + f'OOD_water_{backbone}_eval.npy', allow_pickle=True).item()
ood_embs0['1'] = np.array([dict_[key].squeeze() for key in dict_.keys()])

grouped_embs0 = {}
grouped_embs_train0 = {}

for key in in_data_embs0.keys():
    emb = in_data_embs0[key].squeeze()
    label = key[0]
    place = key[2]
    split = key[4]
    name = f'{label}_{place}'
    
    if split != '0':
        if name not in grouped_embs0.keys():
            grouped_embs0[name] = []
        
        grouped_embs0[name].append(emb)
    else:
        if name not in grouped_embs_train0.keys():
            grouped_embs_train0[name] = []
        
        grouped_embs_train0[name].append(emb)

grouped_embs0 = {name: np.array(grouped_embs0[name]) for name in grouped_embs0.keys()}
grouped_embs_train0 = {name: np.array(grouped_embs_train0[name]) for name in grouped_embs_train0.keys()}



def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


grouped_embs = {name: grouped_embs0[name] for name in grouped_embs0.keys()}
grouped_embs_train = {name: grouped_embs_train0[name] for name in grouped_embs_train0.keys()}
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

def get_nn_probs(dists, alpha=1.):
    probs = np.exp(-1 * alpha * dists)
    return probs

    
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

test_probs = {alpha: [] for alpha in alpha_values}
for key in test_dict.keys():
    
    targets = test_dict[key]
    labels = float(key[0]) * np.ones(len(test_dict[key]))
    
    dists = []
    for c in range(len(train_dict_list)):
        sources = np.concatenate([train_dict_list[c][kk] for kk in train_dict_list[c].keys()])
        dists.append(get_nn_distances(targets, sources))
    dists = np.array(dists)
    knn_accs = []
    for alpha in alpha_values:
        probs = get_nn_probs(dists, alpha).sum(-1)
        preds = np.argmax(probs, axis=0)
        probs = np.min(probs, axis=0)
        test_probs[alpha].append(probs)
        knn_accs.append(accuracy_score(labels, preds))
    
    print(key, np.round(knn_accs, 5))

test_probs = {k: np.concatenate(test_probs[alpha]) for k in alpha_values}

ood_embs = np.concatenate([ood_dict[key] for key in ood_dict.keys()])


ood_probs = {alpha: [] for alpha in alpha_values}

dists = []
for c in range(len(train_dict_list)):
    sources = np.concatenate([train_dict_list[c][kk] for kk in train_dict_list[c].keys()])
    dists.append(get_nn_distances(ood_embs, sources))
dists = np.array(dists)

for alpha in alpha_values:
    probs = get_nn_probs(dists, alpha).sum(-1)
    preds = np.argmax(probs, axis=0)
    probs = np.min(probs, axis=0)
    test_probs[alpha].append(probs)

ood_probs = {alpha: np.concatenate(ood_probs[alpha]) for alpha in alpha_values}


print('OOD:')
for alpha in alpha_values:
    print(backbone, normalize_embs, alpha)
    max_val = max(np.max(test_probs[alpha], np.max(ood_probs[alpha])))
    dist_utils.calc_ROC_with_dists(max_val - test_probs[alpha], max_val - ood_probs[alpha])


