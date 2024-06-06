import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import dist_utils
import os
import warnings
from sklearn.metrics import accuracy_score

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


if normalize_embs:
    train_dict = {key: normalize(grouped_embs_train0[key]) for key in grouped_embs_train0.keys()}
    test_dict = {key: normalize(grouped_embs0[key]) for key in grouped_embs0.keys()}
else:
    train_dict = grouped_embs_train0
    test_dict = grouped_embs




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
    
    


def calculate_covariance_matrix(data):
    mean_vector = np.mean(data, axis=0)
    covariance_matrix = np.cov(data - mean_vector, rowvar=False)
    alpha = 0.01 * np.mean(np.diag(covariance_matrix))
    covariance_matrix = covariance_matrix + alpha * np.eye(data.shape[1])
    return covariance_matrix

def compute_scatter_matrix(prototype1, prototype2):
    mean_prototype = (prototype1 + prototype2) / 2
    scatter_matrix1 = np.outer(prototype1 - mean_prototype, prototype1 - mean_prototype)
    scatter_matrix2 = np.outer(prototype2 - mean_prototype, prototype2 - mean_prototype)
    scatter_matrix = scatter_matrix1 + scatter_matrix2
    return scatter_matrix

    
train_dict_list = get_class_dicts(train_dict)
test_dict_list = get_class_dicts(test_dict)

train_prototypes = []
train_embs = []
for data in train_dict_list:
    all_data = []
    for key in data.keys():
        all_data.append(data[key])
        
    all_data = np.concatenate(all_data)
    train_embs.append(all_data)
    train_prototypes.append(all_data.mean(0))

train_prototypes = np.array(train_prototypes)

for key in test_dict.keys():
    data = test_dict[key]
    label = float(key[:1]) * np.ones(len(data))
    
    dists = np.linalg.norm(data[..., None] - train_prototypes.T[None], axis=1)
    preds = np.argmin(dists, -1)    
    
    acc = np.round(accuracy_score(label, preds), 5)
    print(f'group: {key}, acc: {acc}')
    
    
class_covs = [calculate_covariance_matrix(train_embs[i]) for i in range(2)]



x_test = [test_dict[key] for key in test_dict]
x_test = np.concatenate(x_test)

y_test = [float(key[0]) * np.ones(len(test_dict[key])) for key in test_dict]
y_test = np.concatenate(y_test)

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
    
aug_prototypes = np.array(aug_prototypes)


# sp_scatter1 = compute_scatter_matrix(aug_prototypes[0], aug_prototypes[1])
# sp_scatter2 = compute_scatter_matrix(aug_prototypes[2], aug_prototypes[3])
# core_scatter1 = compute_scatter_matrix(aug_prototypes[0], aug_prototypes[3])
# core_scatter2 = compute_scatter_matrix(aug_prototypes[1], aug_prototypes[2])


sp_scatter1 = compute_scatter_matrix(train_dict['0_0'].mean(0),
                                     train_dict['0_1'].mean(0))
sp_scatter2 = compute_scatter_matrix(train_dict['1_0'].mean(0),
                                     train_dict['1_1'].mean(0))

core_scatter1 = compute_scatter_matrix(train_dict['0_0'].mean(0),
                                     train_dict['1_0'].mean(0))
core_scatter2 = compute_scatter_matrix(train_dict['0_1'].mean(0),
                                     train_dict['1_1'].mean(0))


sp_scatter = sp_scatter1 + sp_scatter2
core_scatter = core_scatter1 + core_scatter2


def LDA_projection(within_class_scatter,
                   between_class_scatter,
                   k=500, regularization=1e-6):
    within_class_scatter += np.eye(within_class_scatter.shape[0]) * regularization
    
    inv_within_class_scatter = np.linalg.inv(within_class_scatter)
    
    sw_inv_sb = np.dot(inv_within_class_scatter, between_class_scatter)
    eigenvalues, eigenvectors = np.linalg.eig(sw_inv_sb)

    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    #sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    projection_matrix = sorted_eigenvectors[:, :k]
    
    return projection_matrix


proj_mat = LDA_projection(sp_scatter, core_scatter)


def project_data(projection_matrix, data):
    transformed_data = np.dot(data, projection_matrix)
    return transformed_data



train_dict = {key: project_data(proj_mat, grouped_embs_train0[key]) for key in grouped_embs_train0.keys()}
test_dict = {key: project_data(proj_mat, grouped_embs0[key]) for key in grouped_embs0.keys()}


train_dict_list = get_class_dicts(train_dict)
test_dict_list = get_class_dicts(test_dict)


train_prototypes = []
train_embs = []
for data in train_dict_list:
    all_data = []
    for key in data.keys():
        all_data.append(data[key])
        
    all_data = np.concatenate(all_data)
    train_embs.append(all_data)
    train_prototypes.append(all_data.mean(0))



train_prototypes = np.array(train_prototypes)

for key in test_dict.keys():
    data = test_dict[key]
    label = float(key[:1]) * np.ones(len(data))
    
    dists = np.linalg.norm(data[..., None] - train_prototypes.T[None], axis=1)
    preds = np.argmin(dists, -1)    
    
    acc = np.round(accuracy_score(label, preds), 5)
    print(f'group: {key}, acc: {acc}')
    
    
    
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
    
aug_prototypes = np.array(aug_prototypes)
