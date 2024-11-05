import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import dist_utils
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

seed = 2
np.random.seed(seed+1)

warnings.filterwarnings("ignore")

normalize_embs = True


backbones = ['dino', 'res50', 'res18']
backbone = backbones[2]
resnet_types = ['pretrained', 'finetuned', 'scratch']
resnet_type = resnet_types[0]


n_c = 5

data_path = 'embeddings_raw/'


train_emb_dict = np.load(data_path + 'cmnist_train_res18_pretrained.npy', allow_pickle=True).item()
val_emb_dict = np.load(data_path + 'cmnist_val_res18_pretrained.npy', allow_pickle=True).item()

train_dict0 = {}
for i in range(n_c):
    for j in range(n_c):
        key = f'{i}_{j}'
        train_dict0[key] = train_emb_dict[key]
        
test_dict0 = {}
for i in range(n_c):
    for j in range(n_c):
        key = f'{i}_{j}'
        test_dict0[key] = val_emb_dict[key]
        
ood_dict0 = {}
for i in range(n_c):
    ood_dict0[f'{i}'] = val_emb_dict[f'{i+5}_{i}']
        


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


if normalize_embs:
    train_dict = {key: normalize(train_dict0[key]) for key in train_dict0.keys()}
    test_dict = {key: normalize(test_dict0[key]) for key in test_dict0.keys()}
    ood_dict = {key: normalize(ood_dict0[key]) for key in ood_dict0.keys()}
else:
    train_dict = train_dict0
    test_dict = test_dict0
    ood_dict = ood_dict0


def get_class_dicts(input_dict):
    class_dicts = []
    for core_name in range(n_c):
        class_dict = {}
        for sp_name in range(n_c):
            name = f'{core_name}_{sp_name}'
            class_dict[name] = input_dict[name]
        class_dicts.append(class_dict)

    return class_dicts


def extract_prototypes(embs, th=0.8):
    embs = np.array(embs)
    prototype = embs.mean(0)
    dists = np.linalg.norm(embs - prototype, axis=-1)
    th_val = np.sort(dists)[int(len(dists) * th)]
    valid_inds = np.argwhere(dists < th_val).ravel()
    final_prototype = embs[valid_inds].mean(0)
    return final_prototype

def refine_group_prototypes(group_embs):
    all_embs = np.concatenate(group_embs)
    print(group_embs[0].shape, group_embs[1].shape)
    first_prototypes = [embs.mean(0) for embs in group_embs]
    first_prototypes = np.array(first_prototypes)
    
    dists = np.linalg.norm(all_embs[..., None] - first_prototypes.T[None], axis=1)
    labels = np.argmin(dists, axis=1)
    new_embs = []
    for l in np.unique(labels):
        inds = np.argwhere(labels == l).ravel()
        new_embs.append(all_embs[inds])

    print(new_embs[0].shape, new_embs[1].shape)   
    print()
    
    new_prototypes = [embs.mean(0) for embs in new_embs]
    
    return new_prototypes
    
    


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

dist_utils.calc_ROC(test_dict, ood_embs, prototypes=train_prototypes, plot=True,
                    exp_name='Prototypical', network_name=network_name)


x_train = np.concatenate(train_embs)
y_train = np.concatenate([c * np.ones(len(train_embs[c])) for c in range(n_c)])


dists = np.linalg.norm(x_train[..., None] - train_prototypes.T[None], axis=1)
y_hat_train = np.argmin(dists, -1)  

accuracy = accuracy_score(y_hat_train, y_train)
print('acc:', accuracy)

print()

total_misc_inds = np.argwhere(y_hat_train != y_train).ravel()
total_crr_inds = np.argwhere(y_hat_train == y_train).ravel()

aug_prototypes = []
aug_embs = []
for l in range(n_c):
    class_inds = np.argwhere(y_train == l).ravel()

    class_crr_inds = np.intersect1d(class_inds, total_crr_inds, assume_unique=True)
    crr_prototype = x_train[class_crr_inds].mean(0)
    aug_prototypes.append(crr_prototype)
    aug_embs.append(x_train[class_crr_inds])

    for j in range(n_c):
        if j == l:
            continue
        
        class_miss_inds = np.intersect1d(class_inds, total_misc_inds, assume_unique=True)
        trg_lbl_inds = np.argwhere(y_hat_train == j).ravel()
        class_miss_inds_trg = np.intersect1d(class_miss_inds, trg_lbl_inds, assume_unique=True)
        
        if len(class_miss_inds_trg) < 1:
            print('empty')
            trg_prototype = crr_prototype.copy()
        else:
            trg_prototype = x_train[class_miss_inds_trg].mean(0)
        aug_prototypes.append(trg_prototype)
        aug_embs.append(x_train[class_miss_inds_trg])
    


aug_prototypes = np.array(aug_prototypes)
print('Prototypical-GI:')
dist_utils.calc_ROC(test_dict, ood_embs, prototypes=aug_prototypes, plot=True,
                    exp_name='Prototypical-GI', network_name=network_name)

# refined_prototypes = []

# for c in range(n_c):
#     refined_prototypes.extend(refine_group_prototypes(aug_embs[n_c*c: n_c*(c+1)]))

# print('after (refined):')
# dist_utils.calc_ROC(test_dict, ood_embs, prototypes=refined_prototypes, plot=False)

aug_prototypes = np.array(aug_prototypes)
aug_prototypes2 = []
for c in range(n_c):
    aug_prototypes2.append(aug_prototypes[n_c*c: n_c*(c+1)].mean())

print('Prototypical-GI-MG:')
dist_utils.calc_ROC(test_dict, ood_embs, prototypes=aug_prototypes2, plot=True,
                    exp_name='Prototypical-GI-MG', network_name=network_name)


# refined_prototypes = np.array(refined_prototypes)
# aug_prototypes2 = []
# for c in range(n_c):
#     aug_prototypes2.append(refined_prototypes[n_c*c: n_c*(c+1)].mean())

# print('Prototypical-GI-MG (refined):')
# dist_utils.calc_ROC(test_dict, ood_embs, prototypes=aug_prototypes2, plot=False,
#                     exp_name='Prototypical-GI-MG (refined)', network_name=network_name)
