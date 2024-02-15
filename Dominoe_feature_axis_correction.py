import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")


seed = 8
np.random.seed(seed)

samples4prototype = 5

grouped_embs0 = np.load('Dominoes_grouped_embs.npy', allow_pickle=True).item()
    
def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def get_prototypes(embeddings, n_data=None):
    
    n = len(embeddings)
    
    if n_data is None:
        inds = np.arange(n)
    else:
        assert n_data < n
        inds = np.random.choice(n, n_data, replace=False)
    
    prototype = embeddings[inds].mean(axis=0, keepdims=True)
    
    return prototype


grouped_embs = {name: normalize(grouped_embs0[name]) for name in grouped_embs0.keys()}
    
grouped_prototypes = {group: get_prototypes(embs, samples4prototype)\
                      for group, embs in grouped_embs.items()}

all_embs = np.concatenate(list(grouped_embs.values()))
all_prototypes = np.concatenate(list(grouped_prototypes.values()))
group_names = list(grouped_embs.keys())


##############################################################



#sp_ax1 = normalize(grouped_prototypes['1_airplane'] - grouped_prototypes['0_airplane'])
#sp_ax2 = normalize(grouped_prototypes['1_car'] - grouped_prototypes['0_car'])
#core_ax1 = normalize(grouped_prototypes['1_car'] - grouped_prototypes['1_airplane'])
#core_ax2 = normalize(grouped_prototypes['0_car'] - grouped_prototypes['0_airplane'])

sp_ax1 = normalize(normalize(grouped_prototypes['1_airplane']) + normalize(grouped_prototypes['1_car']))
sp_ax2 = normalize(normalize(grouped_prototypes['0_airplane']) + normalize(grouped_prototypes['0_car']))
core_ax1 = normalize(normalize(grouped_prototypes['0_airplane']) + normalize(grouped_prototypes['1_airplane']))
core_ax2 = normalize(normalize(grouped_prototypes['0_car']) + normalize(grouped_prototypes['1_car']))



def refine_embs(embs, sp1, sp2, cr1, cr2):
    #core = embs * core_ax_normal[None]
    embs = normalize(embs)
    sp_coefs1 = np.dot(embs, sp1.squeeze())
    sp_coefs2 = np.dot(embs, sp2.squeeze())

    cr_coefs1 = np.dot(embs, cr1.squeeze())
    cr_coefs2 = np.dot(embs, cr2.squeeze())
    
    #refined = embs.copy()
    refined = cr_coefs1[:, None] * np.repeat(cr1, embs.shape[0], axis=0) - sp_coefs1[:, None] * np.repeat(sp1, embs.shape[0], axis=0)
    refined += cr_coefs2[:, None] * np.repeat(cr2, embs.shape[0], axis=0) - sp_coefs2[:, None] * np.repeat(sp2, embs.shape[0], axis=0)
    
    refined = normalize(refined)
    return refined


refined_grouped_embs = {}
for key in grouped_embs.keys():
    refined_grouped_embs[key] = refine_embs(grouped_embs[key], sp_ax1, sp_ax2, core_ax1, core_ax2)

corrupted_grouped_embs = {}
for key in grouped_embs.keys():
    corrupted_grouped_embs[key] = refine_embs(grouped_embs[key], core_ax1, core_ax2, sp_ax1, sp_ax2)


refined_grouped_prototypes = {group: get_prototypes(embs, samples4prototype)\
                              for group, embs in refined_grouped_embs.items()}


##############################################################

def prepare_class_data(embs_dict, group_names, len_g, lbl_val, inds=None):
    data_list = []
    for name in group_names:
        if inds is None:
            g_inds = np.random.choice(len(embs_dict[name]), len_g, replace=False)
        else:
            g_inds = inds
            
        assert len(g_inds) == len_g
        
        data_list.append(embs_dict[name][g_inds])
    
    data = np.concatenate(data_list)
    labels = lbl_val * np.ones(len(group_names) * len_g)
    
    return data, labels

def prepare_data(embs_dict, names_0, names_1, len_g, inds=None):
    data0, lbl0 = prepare_class_data(embs_dict, names_0, len_g, 0, inds)
    data1, lbl1 = prepare_class_data(embs_dict, names_1, len_g, 1, inds)
    
    data_np = np.concatenate([data0, data1])
    lbl_np = np.concatenate([lbl0, lbl1]).astype(int)
    
    return data_np, lbl_np


##############################################################

print('Neutral version:')

x_train, y_train = prepare_data(grouped_embs, ['0_airplane', '0_car'], ['1_airplane', '1_car'], 300, np.arange(300))
clf = LogisticRegression()
clf.fit(x_train, y_train)

x_eval, y_eval = prepare_data(grouped_embs, ['0_airplane', '0_car'], ['1_airplane', '1_car'], 300, np.arange(-300,0))
preds = clf.predict(x_eval)
eval_acc = 100 * (preds == y_eval).mean()

print('ZERO / ONE ACC:', eval_acc)

x_train, y_train = prepare_data(grouped_embs, ['0_airplane', '1_airplane'], ['0_car', '1_car'], 300, np.arange(300))
clf = LogisticRegression()
clf.fit(x_train, y_train)

x_eval, y_eval = prepare_data(grouped_embs, ['0_airplane', '1_airplane'], ['0_car', '1_car'], 300, np.arange(-300,0))
preds = clf.predict(x_eval)
eval_acc = 100 * (preds == y_eval).mean()

print('AIRPLANE / CAR ACC:', eval_acc)


##############################################################

print('Refined version:')

x_train, y_train = prepare_data(grouped_embs, ['0_airplane', '0_car'], ['1_airplane', '1_car'], 300, np.arange(300))
clf = LogisticRegression()
clf.fit(x_train, y_train)

x_eval, y_eval = prepare_data(refined_grouped_embs, ['0_airplane', '0_car'], ['1_airplane', '1_car'], 300, np.arange(-300,0))
preds = clf.predict(x_eval)
eval_acc = 100 * (preds == y_eval).mean()

print('ZERO / ONE ACC:', eval_acc)

x_train, y_train = prepare_data(grouped_embs, ['0_airplane', '1_airplane'], ['0_car', '1_car'], 300, np.arange(300))
clf = LogisticRegression()
clf.fit(x_train, y_train)

x_eval, y_eval = prepare_data(refined_grouped_embs, ['0_airplane', '1_airplane'], ['0_car', '1_car'], 300, np.arange(-300,0))
preds = clf.predict(x_eval)
eval_acc = 100 * (preds == y_eval).mean()

print('AIRPLANE / CAR ACC:', eval_acc)

##############################################################
print('Corrupted version:')

x_train, y_train = prepare_data(grouped_embs, ['0_airplane', '0_car'], ['1_airplane', '1_car'], 300, np.arange(300))
clf = LogisticRegression()
clf.fit(x_train, y_train)

x_eval, y_eval = prepare_data(corrupted_grouped_embs, ['0_airplane', '0_car'], ['1_airplane', '1_car'], 300, np.arange(-300,0))
preds = clf.predict(x_eval)
eval_acc = 100 * (preds == y_eval).mean()

print('ZERO / ONE ACC:', eval_acc)

x_train, y_train = prepare_data(grouped_embs, ['0_airplane', '1_airplane'], ['0_car', '1_car'], 300, np.arange(300))
clf = LogisticRegression()
clf.fit(x_train, y_train)

x_eval, y_eval = prepare_data(corrupted_grouped_embs, ['0_airplane', '1_airplane'], ['0_car', '1_car'], 300, np.arange(-300,0))
preds = clf.predict(x_eval)
eval_acc = 100 * (preds == y_eval).mean()

print('AIRPLANE / CAR ACC:', eval_acc)

##############################################################
print()


def calc_cos_dist(embs, prototypes):
    embs_normalized = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    prototypes_normalized = prototypes / np.linalg.norm(prototypes, axis=-1, keepdims=True)
    cos_dist = (1 - (embs_normalized[:, None] * prototypes_normalized).sum(axis=-1)) / 2
    dist = np.abs(cos_dist.squeeze())
    return dist


grouped_cos_dist = {group: calc_cos_dist(embs, refined_grouped_prototypes[group])\
                    for group, embs in refined_grouped_embs.items()}

ood_class_names = ['ship', 'truck']
selected_groups_names = ['0_airplane', '0_car', '1_airplane', '1_car']
selected_grouped_embs = {name: refined_grouped_embs[name] for name in selected_groups_names}
    
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for group, ax in zip([group for group in selected_grouped_embs], axes):
    sns.histplot(grouped_cos_dist[group], label=group, palette=['red'], ax=ax, element='step', fill=False)
    sp_name = group[:2]
    ood_embs = np.concatenate([refined_grouped_embs[sp_name + class_name] for class_name in ood_class_names])
    sns.histplot(calc_cos_dist(ood_embs, refined_grouped_prototypes[group]), label='ood', ax=ax, element='step', fill=False)
    ax.legend()
    ax.set_title(group)
    
    
grouped_cos_dist = {group: calc_cos_dist(embs, grouped_prototypes[group]) for group, embs in grouped_embs.items()}

ood_class_names = ['ship', 'truck']
selected_groups_names = ['0_airplane', '0_car', '1_airplane', '1_car']
selected_grouped_embs = {name: grouped_embs[name] for name in selected_groups_names}
    
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for group, ax in zip([group for group in selected_grouped_embs], axes):
    sns.histplot(grouped_cos_dist[group], label=group, palette=['red'], ax=ax, element='step', fill=False)
    sp_name = group[:2]
    ood_embs = np.concatenate([grouped_embs[sp_name + class_name] for class_name in ood_class_names])
    sns.histplot(calc_cos_dist(ood_embs, grouped_prototypes[group]), label='ood', ax=ax, element='step', fill=False)
    ax.legend()
    ax.set_title(group)
    
def get_dist_vals(emb_name1, emb_name2, pr_name1, pr_name2, refined=False):
    
    if refined:
        embs = refined_grouped_embs
        protos = refined_grouped_prototypes
    else:
        embs = grouped_embs
        protos = grouped_prototypes
        
    dist_vals = calc_cos_dist(embs[emb_name1 + emb_name2],
                              protos[pr_name1 + pr_name2])
    return dist_vals
    

def find_thresh_val(main_vals, th=0.95):
    thresh = np.sort(main_vals)[int(th * len(main_vals))]
    return thresh
    
    
for ood_name in ['truck', 'ship']:
    for core_name in ['car', 'airplane']:
        print(f'core name: {core_name}, ood name: {ood_name}')

        neutral_ood = np.concatenate([
                get_dist_vals('0_', ood_name, '0_', core_name),
                get_dist_vals('1_', ood_name, '1_', core_name)
                ])

        refined_ood = np.concatenate([
                get_dist_vals('0_', ood_name, '0_', core_name, refined=True),
                get_dist_vals('1_', ood_name, '1_', core_name, refined=True)
                ])
    
        neutral_ind = np.concatenate([
                get_dist_vals('0_', core_name, '0_', core_name),
                get_dist_vals('1_', core_name, '1_', core_name)
                ])

        refined_ind = np.concatenate([
                get_dist_vals('0_', core_name, '0_', core_name, refined=True),
                get_dist_vals('1_', core_name, '1_', core_name, refined=True)
                ])
    
        neutral_sp = np.concatenate([
                get_dist_vals('0_', core_name, '1_', core_name),
                get_dist_vals('1_', core_name, '0_', core_name)
                ])

        refined_sp = np.concatenate([
                get_dist_vals('0_', core_name, '1_', core_name, refined=True),
                get_dist_vals('1_', core_name, '0_', core_name, refined=True)
                ])

    
                
        neutral_th = find_thresh_val(np.concatenate([neutral_ind, neutral_sp]))
        neutral_err = neutral_ood[neutral_ood < neutral_th].shape[0] / neutral_ood.shape[0]

        refined_th = find_thresh_val(np.concatenate([refined_ind, refined_sp]))
        refined_err = refined_ood[refined_ood < refined_th].shape[0] / refined_ood.shape[0]
        
        print('neutral:', neutral_err,
              np.mean(neutral_ind) / np.mean(neutral_ood),
              np.mean(neutral_sp) / np.mean(neutral_ood))
        
        print('refined:', refined_err,
              np.mean(np.mean(refined_ind)) / np.mean(np.mean(refined_ood)),
              np.mean(refined_sp) / np.mean(refined_ood))
        
        print('***********************')
        
        