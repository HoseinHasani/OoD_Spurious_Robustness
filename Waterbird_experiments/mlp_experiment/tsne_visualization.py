import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import dist_utils
import os
import warnings
from sklearn.manifold import TSNE

tsne_perplexity = 120

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


if normalize_embs:
    train_dict = {key: normalize(grouped_embs_train0[key]) for key in grouped_embs_train0.keys()}
    test_dict = {key: normalize(grouped_embs0[key]) for key in grouped_embs0.keys()}
    ood_dict = {key: normalize(ood_embs0[key]) for key in ood_embs0.keys()}
else:
    train_dict = grouped_embs_train0
    test_dict = grouped_embs
    ood_dict = ood_embs0



def plot_tsne(tsne_embs, labels, name, figsize=6):
    
    #cmap = plt.get_cmap('jet')
    #colors = np.random.permutation(K)
    #colors = cmap(colors / np.max(colors) * 1)
    
    colors = ['tab:blue', 'tab:green', 'tab:pink', 'tab:orange', 'tab:gray', 'tab:green']
    
    plt.figure(figsize=(figsize, figsize))
    
    inds = np.argwhere(labels == 0).ravel()
    proto_ind = np.argwhere(labels == 6).ravel()[0]
    
    plt.scatter(tsne_embs[inds][:, 0], tsne_embs[inds][:, 1], marker='.', s=5,
                c=colors[0], label='group 0', alpha=0.4)
    plt.scatter(tsne_embs[proto_ind: proto_ind+1][:, 0],
                tsne_embs[proto_ind: proto_ind+1][:, 1],
                marker='*', s=40, c=colors[0], label='group 0 (prototype)')   
    
    inds = np.argwhere(labels == 1).ravel()
    proto_ind = np.argwhere(labels == 7).ravel()[0]
    
    plt.scatter(tsne_embs[inds][:, 0], tsne_embs[inds][:, 1], marker='.', s=5,
                c=colors[1], label='group 1', alpha=0.4)
    plt.scatter(tsne_embs[proto_ind: proto_ind+1][:, 0],
                tsne_embs[proto_ind: proto_ind+1][:, 1],
                marker='*', s=40, c=colors[1], label='group 1 (prototype)')   
    
      
    inds = np.argwhere(labels == 2).ravel()
    proto_ind = np.argwhere(labels == 8).ravel()[0]
    
    plt.scatter(tsne_embs[inds][:, 0], tsne_embs[inds][:, 1], marker='.', s=5,
                c=colors[2], label='group 2', alpha=0.4)
    plt.scatter(tsne_embs[proto_ind: proto_ind+1][:, 0],
                tsne_embs[proto_ind: proto_ind+1][:, 1],
                marker='*', s=40, c=colors[2], label='group 2 (prototype)')   
    
    
    inds = np.argwhere(labels == 3).ravel()
    proto_ind = np.argwhere(labels == 9).ravel()[0]
    
    plt.scatter(tsne_embs[inds][:, 0], tsne_embs[inds][:, 1], marker='.', s=5,
                c=colors[3], label='group 3', alpha=0.4)
    plt.scatter(tsne_embs[proto_ind: proto_ind+1][:, 0],
                tsne_embs[proto_ind: proto_ind+1][:, 1],
                marker='*', s=40, c=colors[3], label='group 3 (prototype)')   
    
    
    inds = np.concatenate([np.argwhere(labels == 4).ravel(), np.argwhere(labels == 5).ravel()])
    
    plt.scatter(tsne_embs[inds][:, 0], tsne_embs[inds][:, 1], marker='.', s=5,
                c=colors[4], label='OOD', alpha=0.2)
    
    
    
    plt.title(name)
    plt.legend()
    plt.savefig(name + '.png', dpi=130)
    
labels = []
embs = []

embs.append(test_dict['0_0'])
labels.append(0 * np.ones(len(train_dict['0_0'])))

embs.append(test_dict['0_1'])
labels.append(1 * np.ones(len(train_dict['0_1'])))

embs.append(test_dict['1_0'])
labels.append(2 * np.ones(len(train_dict['1_0'])))

embs.append(test_dict['1_1'])
labels.append(3 * np.ones(len(train_dict['1_1'])))

embs.append(ood_dict['0'])
labels.append(4 * np.ones(len(ood_dict['0'])))

embs.append(ood_dict['1'])
labels.append(5 * np.ones(len(ood_dict['1'])))

embs.append(test_dict['0_0'].mean(0, keepdims=True))
labels.append(6 * np.ones(1))

embs.append(test_dict['0_1'].mean(0, keepdims=True))
labels.append(7 * np.ones(1))

embs.append(test_dict['1_0'].mean(0, keepdims=True))
labels.append(8 * np.ones(1))

embs.append(test_dict['1_1'].mean(0, keepdims=True))
labels.append(9 * np.ones(1))




all_embs = np.concatenate(embs)
all_labels = np.concatenate(labels)

tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=tsne_perplexity)
tsne_embs = tsne.fit_transform(all_embs)

plot_tsne(tsne_embs, all_labels, 'DINO t-SNE Embeddings')


