import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import dist_utils
import os
import warnings
from sklearn.cluster import KMeans
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sprod')))
from sprod1 import SPROD1
from sprod2 import SPROD2
from sprod3 import SPROD3
from sprod4 import SPROD4


warnings.filterwarnings("ignore")

normalize_embs = True
normalize_embs2 = True

backbones = ['dino', 'res50', 'res18']
backbone = backbones[1]
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
grouped_embs_val0 = {}

for key in in_data_embs0.keys():
    emb = in_data_embs0[key].squeeze()
    label = key[0]
    place = key[2]
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



def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


grouped_embs = {name: grouped_embs0[name] for name in grouped_embs0.keys()}
grouped_embs_train = {name: grouped_embs_train0[name] for name in grouped_embs_train0.keys()}
ood_embs = {name: ood_embs0[name] for name in ood_embs0.keys()}



train_dict = grouped_embs_train0
test_dict = grouped_embs0
ood_dict = ood_embs0

def normalize2(x, mean, std, eps=1e-7):
    normalized = (x - mean) / (std + eps)
    return normalized


if normalize_embs2:
    
    train_embs = np.concatenate([train_dict[k] for k in train_dict.keys()])
    mean = train_embs.mean(0)
    std = train_embs.std(0)
    
    train_dict = {key: normalize2(train_dict[key], mean, std) for key in train_dict.keys()}
    test_dict = {key: normalize2(test_dict[key], mean, std) for key in test_dict.keys()}
    ood_dict = {key: normalize2(ood_dict[key], mean, std) for key in ood_dict.keys()}
        
    
    
if normalize_embs:
    train_dict = {key: normalize(train_dict[key]) for key in train_dict.keys()}
    test_dict = {key: normalize(test_dict[key]) for key in test_dict.keys()}
    ood_dict = {key: normalize(ood_dict[key]) for key in ood_dict.keys()}




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

def extract_prototypes(embs, th=0.8):
    embs = np.array(embs)
    prototype = embs.mean(0)
    dists = np.linalg.norm(embs - prototype, axis=-1)
    th_val = np.sort(dists)[int(len(dists) * th)]
    valid_inds = np.argwhere(dists < th_val).ravel()
    final_prototype = embs[valid_inds].mean(0)
    return final_prototype

def refine_group_prototypes(group_embs, n_iter=1):
    all_embs = np.concatenate(group_embs)

    prototypes = [embs.mean(0) for embs in group_embs]
    prototypes = np.array(prototypes)
    
    # print([group_embs[j].shape for j in range(len(group_embs))]) 
    
    for k in range(n_iter):
        dists = np.linalg.norm(all_embs[..., None] - prototypes.T[None], axis=1)
        labels = np.argmin(dists, axis=1)
        new_embs = []
        for l in np.unique(labels):
            inds = np.argwhere(labels == l).ravel()
            new_embs.append(all_embs[inds])

        # print([new_embs[j].shape for j in range(len(new_embs))]) 
    
        prototypes = [embs.mean(0) for embs in new_embs]
        prototypes = np.array(prototypes)
    # print()
    
    
    return prototypes
    
    
def cluster_group_prototypes(group_embs, n_iter=100):
    n_c = len(group_embs)
    kmeans = KMeans(n_clusters=n_c, max_iter=n_iter)
    all_embs = np.concatenate(group_embs)
    
    kmeans.fit(all_embs)

    prototypes = kmeans.cluster_centers_
    # print(prototypes.shape)
    
    return prototypes
    
core_ax, sp_ax, core_ax_norm, sp_ax_norm = get_axis(train_dict)
print('ax correlation: ', np.dot(core_ax, sp_ax))


    
print()
dist_utils.calc_dists_ratio(train_dict, ood_dict)
dist_utils.calc_dists_ratio(test_dict, ood_dict)

train_dict_list = get_class_dicts(train_dict)
test_dict_list = get_class_dicts(test_dict)

ood_embs = np.concatenate([ood_dict[key] for key in ood_dict.keys()])

train_prototypes = []
train_embs = []
for data in train_dict_list:
    all_data = []
    for key in data.keys():
        all_data.append(data[key])
        
    all_data = np.concatenate(all_data)
    train_embs.append(all_data)
    train_prototypes.append(all_data.mean(0))


#train_prototypes = [train_dict[key].mean(0) for key in train_dict.keys()]
train_prototypes = np.array(train_prototypes)

print('OOD:')
print('Prototypical:')
#dist_utils.calc_ROC(test_dict_list[0], ood_embs, prototypes=train_dict_list[0])
#dist_utils.calc_ROC(test_dict_list[1], ood_embs, prototypes=train_dict_list[1])

network_name = 'ResNet50' if backbone == 'res50' else 'DINO-v2 (Normalized)'

dist_utils.calc_ROC(test_dict, ood_embs, prototypes=train_prototypes, plot=False,
                    exp_name='Prototypical', network_name=network_name)


x_train = np.concatenate(train_embs)
y_train = np.concatenate([np.zeros(len(train_embs[0])), np.ones(len(train_embs[1]))])


test_embs = np.concatenate([test_dict[k] for k in test_dict.keys()])

class DummyConfig:
    class Postprocessor:
        postprocessor_args = {}
    postprocessor = Postprocessor()
    
    
config = DummyConfig()
sprod1 = SPROD1(config=config, probabilistic_score=False, normalize_features=True)
sprod2 = SPROD2(config=config)
sprod3 = SPROD3(config=config)
sprod4 = SPROD4(config=config)


train_preds, train_confs, _ = sprod1.numpy_inference(x_train, y_train)
test_preds, test_confs, _ = sprod1.numpy_inference(test_embs)
ood_preds, ood_confs, _ = sprod1.numpy_inference(ood_embs)

print('sprod-1:')
dist_utils.calc_ROC_with_dists(-test_confs, -ood_confs)


dists = np.linalg.norm(x_train[..., None] - train_prototypes.T[None], axis=1)
y_hat_train = np.argmin(dists, -1)  


total_misc_inds = np.argwhere(y_hat_train != y_train).ravel()
total_crr_inds = np.argwhere(y_hat_train == y_train).ravel()

aug_prototypes = []
aug_embs = []
for l in [0, 1]:
    class_inds = np.argwhere(y_train == l).ravel()
    class_miss_inds = np.intersect1d(class_inds, total_misc_inds, assume_unique=True)
    aug_prototypes.append(x_train[class_miss_inds].mean(0))
    aug_embs.append(x_train[class_miss_inds])
    class_crr_inds = np.intersect1d(class_inds, total_crr_inds, assume_unique=True)
    aug_prototypes.append(x_train[class_crr_inds].mean(0))
    aug_embs.append(x_train[class_crr_inds])
    
refined_prototypes = []

refined_prototypes.extend(refine_group_prototypes(aug_embs[:2]))
refined_prototypes.extend(refine_group_prototypes(aug_embs[2:]))

aug_prototypes = np.array(aug_prototypes)
print('Prototypical-GI:')
dist_utils.calc_ROC(test_dict, ood_embs, prototypes=aug_prototypes, plot=False,
                    exp_name='Prototypical-GI', network_name=network_name)


print('sprod-2:')
train_preds, train_confs, _ = sprod2.numpy_inference2(x_train, y_train)
test_preds, test_confs, _ = sprod2.numpy_inference2(test_embs)
ood_preds, ood_confs, _ = sprod2.numpy_inference2(ood_embs)
dist_utils.calc_ROC_with_dists(-test_confs, -ood_confs)


print('after (refined):')
dist_utils.calc_ROC(test_dict, ood_embs, prototypes=refined_prototypes, plot=False)


print('sprod-3:')
train_preds, train_confs, _ = sprod3.numpy_inference2(x_train, y_train)
test_preds, test_confs, _ = sprod3.numpy_inference2(test_embs)
ood_preds, ood_confs, _ = sprod3.numpy_inference2(ood_embs)
dist_utils.calc_ROC_with_dists(-test_confs, -ood_confs)



aug_prototypes = np.array(aug_prototypes)
# aug_prototypes = np.array(refined_prototypes)
aug_prototypes2 = [aug_prototypes[:2].mean(0), aug_prototypes[2:].mean(0)]
print('Prototypical-GI-MG:')
dist_utils.calc_ROC(test_dict, ood_embs, prototypes=aug_prototypes2, plot=False,
                    exp_name='Prototypical-GI-MG', network_name=network_name)



print('sprod-4:')
train_preds, train_confs, _ = sprod4.numpy_inference2(x_train, y_train)
test_preds, test_confs, _ = sprod4.numpy_inference2(test_embs)
ood_preds, ood_confs, _ = sprod4.numpy_inference2(ood_embs)
dist_utils.calc_ROC_with_dists(-test_confs, -ood_confs)




if False:
    
    n_c = 2
    
    kmeans_prototypes = []
    for c in range(n_c):
        kmeans_prototypes.extend(cluster_group_prototypes(aug_embs[n_c*c: n_c*(c+1)]))
        
    print('Prototypical-KMEANS:')
    dist_utils.calc_ROC(test_dict, ood_embs, prototypes=kmeans_prototypes, plot=False,
                        exp_name='Prototypical-KMEANS', network_name=network_name)
    
    
    
    
    # aug_prototypes = np.concatenate([aug_prototypes, train_prototypes])
    # print('after after:')
    # dist_utils.calc_ROC(test_dict, ood_embs, prototypes=aug_prototypes, plot=True)
    
    # inds1 = [0, 1, 4]
    # inds2 = [2, 3, 5]
    
    # aug_prototypes2 = [aug_prototypes[inds1].mean(0), aug_prototypes[inds2].mean(0)]
    # print('after after2:')
    # dist_utils.calc_ROC(test_dict, ood_embs, prototypes=aug_prototypes2, plot=True)
    
    
    
    n_cluster = 4
    kmeans_protos = []
    
    def propose_centers(embeddings):
        centers = []
        for embs in embeddings:
            for j in range(n_cluster):
                mean = embs.mean(0)
                std = embs.std(0)
                if j == 0:
                    centers.append(mean)
                else:
                    centers.append(np.random.normal(mean, std / 0.2))
        return centers
    
    class1_protos = propose_centers(aug_embs[:2])
    kmeans = KMeans(n_clusters=n_cluster*2, random_state=0, init=class1_protos, max_iter=15).fit(train_embs[0])
    kmeans_protos.append(kmeans.cluster_centers_)
    
    class2_protos = propose_centers(aug_embs[2:])
    kmeans = KMeans(n_clusters=n_cluster*2, random_state=0, init=class2_protos, max_iter=15).fit(train_embs[1])
    kmeans_protos.append(kmeans.cluster_centers_)
    
    kmeans_protos = np.concatenate(kmeans_protos)
    aug_prototypes = np.concatenate([aug_prototypes, kmeans_protos])
    # print('kmeans:')
    # dist_utils.calc_ROC(test_dict, ood_embs, prototypes=aug_prototypes)
    
    
