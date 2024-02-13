import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")


seed = 8
np.random.seed(seed)


grouped_embs = np.load('Dominoes_grouped_embs.npy', allow_pickle=True).item()
    
    
grouped_prototypes = {group: embs.mean(axis=0, keepdims=True) for group, embs in grouped_embs.items()}
all_embs = np.concatenate(list(grouped_embs.values()))
all_prototypes = np.concatenate(list(grouped_prototypes.values()))
group_names = list(grouped_embs.keys())


def normalize(x):
    return x / np.linalg.norm(x)

sp_ax1 = normalize(grouped_prototypes['1_airplane'] - grouped_prototypes['0_airplane'])
sp_ax2 = normalize(grouped_prototypes['1_car'] - grouped_prototypes['0_car'])
core_ax1 = normalize(grouped_prototypes['1_car'] - grouped_prototypes['1_airplane'])
core_ax2 = normalize(grouped_prototypes['0_car'] - grouped_prototypes['0_airplane'])


def refine_embs(embs):
    #core = embs * core_ax_normal[None]
    sp_coefs1 = np.dot(embs, sp_ax1.squeeze())
    sp_coefs2 = np.dot(embs, sp_ax2.squeeze())

    refined = embs.copy()
    refined = refined - sp_coefs1[:, None] * np.repeat(sp_ax1, embs.shape[0], axis=0)
    refined = refined - sp_coefs2[:, None] * np.repeat(sp_ax2, embs.shape[0], axis=0)

    return refined


def calc_cos_dist(embs, prototypes):
    embs_normalized = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    prototypes_normalized = prototypes / np.linalg.norm(prototypes, axis=-1, keepdims=True)
    cos_dist = (1 - (embs_normalized[:, None] * prototypes_normalized).sum(axis=-1)) / 2
    return cos_dist.squeeze()

refined_grouped_embs = {}
for key in grouped_embs.keys():
    refined_grouped_embs[key] = refine_embs(grouped_embs[key])

refined_grouped_prototypes = {}
for key in grouped_prototypes.keys():
    refined_grouped_prototypes[key] = refine_embs(grouped_prototypes[key])

grouped_cos_dist = {group: calc_cos_dist(embs, refined_grouped_prototypes[group]) for group, embs in refined_grouped_embs.items()}

ood_class_names = ['ship', 'truck']
selected_groups_names = ['0_airplane', '0_car', '1_airplane', '1_car']
selected_grouped_embs = {name: refined_grouped_embs[name] for name in selected_groups_names}
    
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for group, ax in zip([group for group in selected_grouped_embs], axes):
    sns.kdeplot(grouped_cos_dist[group], label=group, palette=['red'], ax=ax)
    sp_name = group[:2]
    ood_embs = np.concatenate([refined_grouped_embs[sp_name + class_name] for class_name in ood_class_names])
    sns.kdeplot(calc_cos_dist(ood_embs, refined_grouped_prototypes[group]), label='ood', ax=ax)
    ax.legend()
    ax.set_title(group)
    
    
grouped_cos_dist = {group: calc_cos_dist(embs, grouped_prototypes[group]) for group, embs in grouped_embs.items()}

ood_class_names = ['ship', 'truck']
selected_groups_names = ['0_airplane', '0_car', '1_airplane', '1_car']
selected_grouped_embs = {name: grouped_embs[name] for name in selected_groups_names}
    
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for group, ax in zip([group for group in selected_grouped_embs], axes):
    sns.kdeplot(grouped_cos_dist[group], label=group, palette=['red'], ax=ax)
    sp_name = group[:2]
    ood_embs = np.concatenate([grouped_embs[sp_name + class_name] for class_name in ood_class_names])
    sns.kdeplot(calc_cos_dist(ood_embs, grouped_prototypes[group]), label='ood', ax=ax)
    ax.legend()
    ax.set_title(group)