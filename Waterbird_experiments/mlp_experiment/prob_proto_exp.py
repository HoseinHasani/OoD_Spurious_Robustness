import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import multivariate_normal
import dist_utils
import os
import warnings

warnings.filterwarnings("ignore")

normalize_embs = True


backbones = ['dino', 'res50', 'res18']
backbone = backbones[0]
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




def get_axis(embeddings):
    
    core_ax1 = embeddings[f'{core_class_names[1]}_{sp_class_names[0]}'].mean(0, keepdims=False) - \
                         embeddings[f'{core_class_names[0]}_{sp_class_names[0]}'].mean(0, keepdims=False)
    core_ax2 = embeddings[f'{core_class_names[1]}_{sp_class_names[1]}'].mean(0, keepdims=False) - \
                         embeddings[f'{core_class_names[0]}_{sp_class_names[1]}'].mean(0, keepdims=False)
    core_ax = 0.5 * core_ax1 + 0.5 * core_ax2
    
    sp_ax1 = embeddings[f'{core_class_names[0]}_{sp_class_names[1]}'].mean(0, keepdims=False) - \
                         embeddings[f'{core_class_names[0]}_{sp_class_names[0]}'].mean(0, keepdims=False)
    sp_ax2 = embeddings[f'{core_class_names[1]}_{sp_class_names[1]}'].mean(0, keepdims=False) - \
                         embeddings[f'{core_class_names[1]}_{sp_class_names[0]}'].mean(0, keepdims=False)
    sp_ax = 0.5 * sp_ax1 + 0.5 * sp_ax2
    
    print('axis ratio:', np.linalg.norm(core_ax) / np.linalg.norm(sp_ax))
    
    core_ax_norm = 0.5 * normalize(core_ax1) + 0.5 * normalize(core_ax2)
    sp_ax_norm = 0.5 * normalize(sp_ax1) + 0.5 * normalize(sp_ax2)
    
    return core_ax, sp_ax, core_ax_norm, sp_ax_norm



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

    
core_ax, sp_ax, core_ax_norm, sp_ax_norm = get_axis(train_dict)
print('ax correlation: ', np.dot(core_ax, sp_ax))





def calculate_probabilities(embs, means, stds):
    N, feat_dim = embs.shape
    C, _ = means.shape
    
    if np.isscalar(stds) or stds.ndim == 1:
        # If stds is scalar or 1D, broadcast it to match means shape
        stds = np.broadcast_to(stds.T, (feat_dim, C)).T

    mvns = [multivariate_normal(mean=means[i], cov=np.diag(stds[i] ** 2)) for i in range(C)]
    
    pdfs = np.array([mvn.pdf(embs) for mvn in mvns])
    #probabilities = pdfs.sum(axis=0)
    
    return pdfs


    
print()
dist_utils.calc_dists_ratio(train_dict, ood_dict)
dist_utils.calc_dists_ratio(test_dict, ood_dict)

train_dict_list = get_class_dicts(train_dict)
test_dict_list = get_class_dicts(test_dict)

ood_embs = np.concatenate([ood_dict[key] for key in ood_dict.keys()])

train_prototypes = []
train_stds = []
train_std_vecs = []
train_embs = []
for data in train_dict_list:
    all_data = []
    for key in data.keys():
        all_data.append(data[key])
        
    all_data = np.concatenate(all_data)
    train_embs.append(all_data)
    train_prototypes.append(all_data.mean(0))
    train_stds.append(all_data.std(0).mean())
    train_std_vecs.append(all_data.std(0))
    

train_prototypes = np.array(train_prototypes)
train_stds = 8*np.array(train_stds)
train_std_vecs = 8*np.array(train_std_vecs)

test_embs = np.concatenate([test_dict[key] for key in test_dict.keys()])

print('simple prototypical:')
dist_utils.calc_ROC(test_dict, ood_embs, prototypes=train_prototypes, plot=False)

ind_probs = calculate_probabilities(test_embs, train_prototypes, train_stds).ravel()
ood_probs = calculate_probabilities(ood_embs, train_prototypes, train_stds).ravel()
max_val = max(np.max(ind_probs), np.max(ood_probs))

print('prototypical with scalar std:')
dist_utils.calc_ROC_with_dists(max_val - ind_probs, max_val - ood_probs)



ind_probs = calculate_probabilities(test_embs, train_prototypes, train_std_vecs).ravel()
ood_probs = calculate_probabilities(ood_embs, train_prototypes, train_std_vecs).ravel()
max_val = max(np.max(ind_probs), np.max(ood_probs))

print('prototypical with scalar std:')
dist_utils.calc_ROC_with_dists(max_val - ind_probs, max_val - ood_probs)



x_train = np.concatenate(train_embs)
y_train = np.concatenate([np.zeros(len(train_embs[0])), np.ones(len(train_embs[1]))])


dists = np.linalg.norm(x_train[..., None] - train_prototypes.T[None], axis=1)
y_hat_train = np.argmin(dists, -1)  


total_misc_inds = np.argwhere(y_hat_train != y_train).ravel()
total_crr_inds = np.argwhere(y_hat_train == y_train).ravel()

aug_prototypes = []

for l in [0, 1]:
    class_inds = np.argwhere(y_train == l).ravel()
    class_miss_inds = np.intersect1d(class_inds, total_misc_inds, assume_unique=True)
    aug_prototypes.append(x_train[class_miss_inds].mean(0))
    class_crr_inds = np.intersect1d(class_inds, total_crr_inds, assume_unique=True)
    aug_prototypes.append(x_train[class_crr_inds].mean(0))
    
    
aug_prototypes = np.array(aug_prototypes)
print('after:')
dist_utils.calc_ROC(test_dict, ood_embs, prototypes=aug_prototypes, plot=False)

aug_prototypes = np.concatenate([aug_prototypes, train_prototypes])
print('after after:')
dist_utils.calc_ROC(test_dict, ood_embs, prototypes=aug_prototypes, plot=False)




