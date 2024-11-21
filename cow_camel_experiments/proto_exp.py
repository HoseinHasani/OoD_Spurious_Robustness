import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import dist_utils
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

seed = 0
np.random.seed(seed+1)

warnings.filterwarnings("ignore")

normalize_embs = True




backbones = ['dino', 'res50', 'res18']
backbone = backbones[1]
resnet_types = ['pretrained', 'finetuned', 'scratch']
resnet_type = resnet_types[0]

core_class_names = ['0', '1']
ood_class_names = ['0', '1']
sp_class_names = ['0', '1']

data_path = 'embeddings/'



if backbone == 'dino':
    embeddings_dict = np.load(data_path + 'waterbird_embs_DINO.npy', allow_pickle=True).item()
else:
    embeddings_dict = np.load(data_path + f'cc_embs_{backbone}_{resnet_type}.npy', allow_pickle=True).item()

ood_embs0 = {}
ood_embs0['0'] = embeddings_dict['horse-grass']
ood_embs0['1'] = embeddings_dict['horse-desert']

# # 50% correlation:
# majority_groups = {
#     'camel-desert': 300,  
#     'cow-grass': 300      
# }
# minority_groups = {
#     'camel-grass': 300,    
#     'cow-desert': 300      
# }

# 90% correlation:
majority_groups = {
    'camel-desert': 600,  
    'cow-grass': 600      
}
minority_groups = {
    'camel-grass': 60,    
    'cow-desert': 60      
}

# 95% correlation:
# majority_groups = {
#     'camel-desert': 800,  
#     'cow-grass': 800      
# }
# minority_groups = {
#     'camel-grass': 40,    
#     'cow-desert': 40      
# }

train_dict = {'camel-desert': [], 'camel-grass': [], 'cow-desert': [], 'cow-grass': []}
test_dict = {'camel-desert': [], 'camel-grass': [], 'cow-desert': [], 'cow-grass': []}

def stratified_sampling(group_name, num_train, embeddings):
    train_embs, test_embs = train_test_split(embeddings, train_size=num_train, random_state=seed)
    return train_embs, test_embs

for group_name, num_train in majority_groups.items():
    embeddings = embeddings_dict[group_name]
    
    train_embs, test_embs = stratified_sampling(group_name, num_train, embeddings)
    
    train_dict[group_name] = train_embs
    test_dict[group_name] = test_embs

for group_name, num_train in minority_groups.items():
    embeddings = embeddings_dict[group_name]
    
    train_embs, test_embs = stratified_sampling(group_name, num_train, embeddings)
    
    train_dict[group_name] = train_embs
    test_dict[group_name] = test_embs

train_dict0 = {}
test_dict0 = {}
mapping = {'cow-grass': '0_0', 'cow-desert': '0_1',
           'camel-grass': '1_0', 'camel-desert': '1_1'}

for group_name in train_dict:
    train_dict0[mapping[group_name]] = np.array(train_dict[group_name])
    test_dict0[mapping[group_name]] = np.array(test_dict[group_name])
    


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)



if normalize_embs:
    train_dict = {key: normalize(train_dict0[key]) for key in train_dict0.keys()}
    test_dict = {key: normalize(test_dict0[key]) for key in test_dict0.keys()}
    ood_dict = {key: normalize(ood_embs0[key]) for key in ood_embs0.keys()}
else:
    train_dict = train_dict0
    test_dict = test_dict0
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

    prototypes = [embs.mean(0) for embs in group_embs]
    prototypes = np.array(prototypes)
    
    print(group_embs[0].shape, group_embs[1].shape) 
    
    for k in range(4):
        dists = np.linalg.norm(all_embs[..., None] - prototypes.T[None], axis=1)
        labels = np.argmin(dists, axis=1)
        new_embs = []
        for l in np.unique(labels):
            inds = np.argwhere(labels == l).ravel()
            new_embs.append(all_embs[inds])

        print(new_embs[0].shape, new_embs[1].shape)   
        
        prototypes = [embs.mean(0) for embs in new_embs]
        prototypes = np.array(prototypes)
    print()
    
    
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

print('after (refined):')
dist_utils.calc_ROC(test_dict, ood_embs, prototypes=refined_prototypes, plot=False)

aug_prototypes = np.array(aug_prototypes)
aug_prototypes2 = [aug_prototypes[:2].mean(0), aug_prototypes[2:].mean(0)]
print('Prototypical-GI-MG:')
dist_utils.calc_ROC(test_dict, ood_embs, prototypes=aug_prototypes2, plot=False,
                    exp_name='Prototypical-GI-MG', network_name=network_name)


n_c = 2

kmeans_prototypes = []
for c in range(n_c):
    kmeans_prototypes.extend(cluster_group_prototypes(aug_embs[n_c*c: n_c*(c+1)]))
    
print('Prototypical-KMEANS:')
dist_utils.calc_ROC(test_dict, ood_embs, prototypes=kmeans_prototypes, plot=False,
                    exp_name='Prototypical-KMEANS', network_name=network_name)

