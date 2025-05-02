import numpy as np
import matplotlib.pyplot as plt
import dist_utils
import os
import warnings
import pickle
warnings.filterwarnings("ignore")

n_g_sample_val = 3

normalize_embs = True

core_class_names = ['0', '1']
ood_class_names = ['0', '1']
sp_class_names = ['0', '1']

with open('../grouped_embs.pkl', 'rb') as f:
    grouped_embs0 = pickle.load(f)
    
names = ['woman_black', 'woman_blond', 'man_black', 'man_blond', 'man_bald', 'woman_bald']

grouped_embs1 = {name: np.array(grouped_embs0[name]) for name in grouped_embs0.keys()}

map_dict = {
            'man_black': '0_0',
            'woman_black': '0_1',
            'man_blond': '1_0',     
            'woman_blond': '1_1',     
            }

grouped_embs0 = {map_dict[name]: grouped_embs1[name] for name in names[:-2]}

ood_embs0 = {name: grouped_embs1[name] for name in names[-2:]}


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

    
def refine_group_prototypes(group_embs, prototypes=None, n_iter=2, print_logs=False):
    all_embs = np.concatenate(group_embs)

    if prototypes is None:
        prototypes = [embs.mean(0) for embs in group_embs]
        prototypes = np.array(prototypes)
    
    if print_logs:
        print([group_embs[j].shape for j in range(len(group_embs))]) 
    
    for k in range(n_iter):
        dists = np.linalg.norm(all_embs[..., None] - prototypes.T[None], axis=1)
        labels = np.argmin(dists, axis=1)
        new_embs = []
        for l in np.unique(labels):
            inds = np.argwhere(labels == l).ravel()
            new_embs.append(all_embs[inds])

        if print_logs:
            print([new_embs[j].shape for j in range(len(new_embs))]) 
    
        prototypes = [embs.mean(0) for embs in new_embs]
        prototypes = np.array(prototypes)
        
    if print_logs:
        print()
    
    
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
#dist_utils.calc_ROC(test_dict_list[0], ood_embs, prototypes=train_dict_list[0])
#dist_utils.calc_ROC(test_dict_list[1], ood_embs, prototypes=train_dict_list[1])
dist_utils.calc_ROC(test_dict, ood_embs, prototypes=train_prototypes)


x_train = np.concatenate(train_embs)
y_train = np.concatenate([np.zeros(len(train_embs[0])), np.ones(len(train_embs[1]))])
dists = np.linalg.norm(x_train[..., None] - train_prototypes.T[None], axis=1)
y_hat_train = np.argmin(dists, -1)  


total_miss_inds = np.argwhere(y_hat_train != y_train).ravel()
total_crr_inds = np.argwhere(y_hat_train == y_train).ravel()

aug_prototypes = []
aug_embs = []
for l in [0, 1]:
    class_inds = np.argwhere(y_train == l).ravel()
    class_miss_inds = np.intersect1d(class_inds, total_miss_inds, assume_unique=True)
    aug_prototypes.append(x_train[class_miss_inds].mean(0))
    aug_embs.append(x_train[class_miss_inds])
    
    class_crr_inds = np.intersect1d(class_inds, total_crr_inds, assume_unique=True)
    aug_prototypes.append(x_train[class_crr_inds].mean(0))
    aug_embs.append(x_train[class_crr_inds])
    
aug_prototypes = np.array(aug_prototypes)

dist_utils.calc_ROC(test_dict, ood_embs, prototypes=aug_prototypes)


refined_prototypes = []

refined_prototypes.extend(refine_group_prototypes(aug_embs[:2]))
refined_prototypes.extend(refine_group_prototypes(aug_embs[2:]))

aug_prototypes = np.array(aug_prototypes)

dist_utils.calc_ROC(test_dict, ood_embs, prototypes=refined_prototypes)


#### val

print(f'val {n_g_sample_val} - stage2:')

val_prototypes = []

for k in sorted(train_dict.keys()):
    g_samples = train_dict[k]
    inds = np.random.choice(len(g_samples), n_g_sample_val, replace=False)
    val_prototypes.append(g_samples[inds].mean(0))
    
val_prototypes = np.array(val_prototypes)
dist_utils.calc_ROC(test_dict, ood_embs, prototypes=val_prototypes, plot=False)


refined_val_prototypes = []
refined_val_prototypes.extend(refine_group_prototypes(aug_embs[:2], val_prototypes[:2]))
refined_val_prototypes.extend(refine_group_prototypes(aug_embs[2:], val_prototypes[2:]))
refined_val_prototypes = np.array(refined_val_prototypes)
print(f'val {n_g_sample_val} - stage3:')
dist_utils.calc_ROC(test_dict, ood_embs, prototypes=refined_val_prototypes, plot=False)


