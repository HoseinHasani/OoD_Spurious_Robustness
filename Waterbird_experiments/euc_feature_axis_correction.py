import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import wasserstein_distance
from sklearn.metrics import auc
import seaborn as sns
import os
import warnings
from scipy import integrate

warnings.filterwarnings("ignore")
sns.set_context("paper", font_scale=1.4)     

core_ax_th = 7 # 0.05
sp_ax_th = 5 # 0.15

seed = 8
np.random.seed(seed)

samples4prototype = 400

filter_ood = False

backbones = ['dino', 'res50', 'res18']
backbone = backbones[2]
resnet_types = ['pretrained', 'finetuned', 'scratch']
resnet_type = resnet_types[1]

core_class_names = ['0', '1']
ood_class_names = ['0', '1']
sp_class_names = ['0', '1']
place_names = ['land', 'water']

data_path = 'embeddings/'


if backbone == 'dino':
    in_data_embs0 = np.load(data_path + 'waterbird_embs.npy', allow_pickle=True).item()
elif backbone == 'res50':
    in_data_embs0 = np.load(data_path + f'wb_embs_{backbone}_{resnet_type}.npy', allow_pickle=True).item()

ood_embs0 = {}
if backbone == 'dino':
    dict_ = np.load(data_path + 'land.npy', allow_pickle=True).item()
elif backbone == 'res50':
    dict_ = np.load(data_path + f'land_{backbone}_{resnet_type}.npy', allow_pickle=True).item()
    
ood_embs0['0'] = np.array([dict_[key].squeeze() for key in dict_.keys()])

if backbone == 'dino':
    dict_ = np.load(data_path + 'water.npy', allow_pickle=True).item()
elif backbone == 'res50':
    dict_ = np.load(data_path + f'water_{backbone}_{resnet_type}.npy', allow_pickle=True).item()
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

def get_prototypes(embeddings, n_data=None):
    
    n = len(embeddings)
    
    if n_data is None:
        inds = np.arange(n)
    else:
        assert n_data <= n
        inds = np.random.choice(n, n_data, replace=False)
    
    prototype = embeddings[inds].mean(axis=0, keepdims=True)
    
    return prototype


grouped_embs = {name: np.concatenate([grouped_embs0[name], grouped_embs_train0[name]]) for name in grouped_embs0.keys()}
grouped_embs_train = {name: grouped_embs_train0[name] for name in grouped_embs_train0.keys()}
ood_embs = {name: ood_embs0[name] for name in ood_embs0.keys()}
    
grouped_prototypes = {group: get_prototypes(embs)\
                      for group, embs in grouped_embs_train.items()}

group_names = list(grouped_embs.keys())

##############################################################



#core_ax1 = normalize(normalize(grouped_prototypes[f'{core_class_names[0]}_{sp_class_names[0]}'])\
#                   + normalize(grouped_prototypes[f'{core_class_names[0]}_{sp_class_names[1]}']))
#
#core_ax2 = normalize(normalize(grouped_prototypes[f'{core_class_names[1]}_{sp_class_names[0]}'])\
#                   + normalize(grouped_prototypes[f'{core_class_names[1]}_{sp_class_names[1]}']))
#
#
#
#sp_ax1 = normalize(normalize(grouped_prototypes[f'{core_class_names[0]}_{sp_class_names[0]}'])\
#                   + normalize(grouped_prototypes[f'{core_class_names[1]}_{sp_class_names[0]}'])\
#                   - core_ax1 - core_ax2)
#                   
#    
#sp_ax2 = normalize(normalize(grouped_prototypes[f'{core_class_names[0]}_{sp_class_names[1]}'])\
#                   + normalize(grouped_prototypes[f'{core_class_names[1]}_{sp_class_names[1]}'])\
#                   - core_ax1 - core_ax2)
#
#core_ax1 = normalize(normalize(grouped_prototypes[f'{core_class_names[0]}_{sp_class_names[0]}'])\
#                   + normalize(grouped_prototypes[f'{core_class_names[0]}_{sp_class_names[1]}'])\
#                   - sp_ax1 - sp_ax2)
#    
#core_ax2 = normalize(normalize(grouped_prototypes[f'{core_class_names[1]}_{sp_class_names[0]}'])\
#                   + normalize(grouped_prototypes[f'{core_class_names[1]}_{sp_class_names[1]}'])\
#                   - sp_ax1 - sp_ax2)


core_ax1 = normalize(normalize(grouped_prototypes[f'{core_class_names[1]}_{sp_class_names[0]}'])\
                   - normalize(grouped_prototypes[f'{core_class_names[0]}_{sp_class_names[0]}']))

core_ax2 = normalize(normalize(grouped_prototypes[f'{core_class_names[1]}_{sp_class_names[1]}'])\
                   - normalize(grouped_prototypes[f'{core_class_names[0]}_{sp_class_names[1]}']))



sp_ax1 = normalize(normalize(grouped_prototypes[f'{core_class_names[0]}_{sp_class_names[1]}'])\
                   - normalize(grouped_prototypes[f'{core_class_names[0]}_{sp_class_names[0]}']))

sp_ax2 = normalize(normalize(grouped_prototypes[f'{core_class_names[1]}_{sp_class_names[1]}'])\
                   - normalize(grouped_prototypes[f'{core_class_names[1]}_{sp_class_names[0]}']))


sp_coefs1 = np.dot(sp_ax1, core_ax1.squeeze())
sp_ax1 = sp_ax1 - sp_coefs1 * core_ax1
sp_ax1 = normalize(sp_ax1)

sp_coefs2 = np.dot(sp_ax2, core_ax2.squeeze())
sp_ax2 = sp_ax2 - sp_coefs2 * core_ax2
sp_ax2 = normalize(sp_ax2)

ood_ax1 = normalize(normalize(ood_embs[ood_class_names[0]].mean(axis=0, keepdims=True)))
ood_ax2 = normalize(normalize(ood_embs[ood_class_names[1]].mean(axis=0, keepdims=True)))
    


print('***********************')


print('sp - core:')

print(np.dot(sp_ax1[0], core_ax1[0]))    
print(np.dot(sp_ax2[0], core_ax1[0]))
print(np.dot(sp_ax1[0], core_ax2[0]))    
print(np.dot(sp_ax2[0], core_ax2[0]))

print('sp - ood:')

print(np.dot(sp_ax1[0], ood_ax1[0]))    
print(np.dot(sp_ax2[0], ood_ax1[0]))
print(np.dot(sp_ax1[0], ood_ax2[0]))    
print(np.dot(sp_ax2[0], ood_ax2[0]))

print('core - ood:')

print(np.dot(core_ax1[0], ood_ax1[0]))    
print(np.dot(core_ax1[0], ood_ax1[0]))
print(np.dot(core_ax2[0], ood_ax2[0]))    
print(np.dot(core_ax2[0], ood_ax2[0]))

print('***********************')

##############################################################


def calc_stats(data, cr_ax, sp_ax):
    
    cr_coefs = []
    sp_coefs = []
    for key in data.keys():
        cr_coefs.append(np.abs(np.dot(data[key], cr_ax.squeeze())))
        sp_coefs.append(np.abs(np.dot(data[key], sp_ax.squeeze())))
    
    return np.concatenate(cr_coefs), np.concatenate(sp_coefs)

def calc_CDF(data):
    h, x = np.histogram(data, bins=1000, normed=True)
    F = np.cumsum(h) * (x[1] - x[0])
    return F, x[1:]


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def calc_nonlin_coefs(coefs, F, x, alpha=1):
    f = integrate.cumtrapz(1 - F ** (0.9))
    f = f / f.max() * alpha
    coef_vals = []
    for coef in coefs:
        
        if np.abs(coef) > x.max():
            coef_vals.append(alpha * coef / np.abs(coef))
        else:
            ind = find_nearest(x[:-1], np.abs(coef))
            coef_vals.append(f[ind] * coef / np.abs(coef) * alpha)
            
    return np.array(coef_vals)


c_vals, s_vals = calc_stats(grouped_embs, 0.5 * core_ax1 + 0.5 * core_ax2, 0.5 * sp_ax1 + 0.5 * sp_ax2)
f_c, x_c = calc_CDF(c_vals)
f_s, x_s = calc_CDF(s_vals)

c_vals_ood, s_vals_ood = calc_stats(ood_embs, 0.5 * core_ax1 + 0.5 * core_ax2, 0.5 * sp_ax1 + 0.5 * sp_ax2)
f_c_ood, x_c_ood = calc_CDF(c_vals_ood)
f_s_ood, x_s_ood = calc_CDF(s_vals_ood)

plt.figure()
plt.hist(c_vals, 50, histtype='step', normed=True, linewidth=2.5, label='embs')
plt.hist(c_vals_ood, 50, histtype='step', normed=True, linewidth=2.5, label='ood')
plt.title('core alignment')
plt.legend()

plt.figure()
plt.hist(s_vals, 50, histtype='step', normed=True, linewidth=2.5, label='embs')
plt.hist(s_vals_ood, 50, histtype='step', normed=True, linewidth=2.5, label='ood')
plt.title('sp alignment')
plt.legend()

##############################################################

def refine_embs(embs, sp1, sp2, cr1, cr2, alpha=3., beta=0.1):


    
    
    refined = 1.0 * embs.copy()
    
    cr_coefs1 = np.dot(embs, cr1.squeeze())
    #cr_coefs1 = alpha * np.clip(cr_coefs1, -core_ax_th, core_ax_th)
    cr_coefs1 = calc_nonlin_coefs(cr_coefs1, f_c, x_c, alpha=alpha)
    refined += cr_coefs1[:, None] * np.repeat(cr1, embs.shape[0], axis=0)
    
    
    cr_coefs2 = np.dot(embs, cr2.squeeze())
    #cr_coefs2 = alpha * np.clip(cr_coefs2, -core_ax_th, core_ax_th)
    cr_coefs2 = calc_nonlin_coefs(cr_coefs2, f_c, x_c, alpha=alpha)
    refined += cr_coefs2[:, None] * np.repeat(cr2, embs.shape[0], axis=0)


    sp_coefs1 = np.dot(refined, sp1.squeeze())
    #sp_coefs1 = beta * np.clip(sp_coefs1, -sp_ax_th, sp_ax_th)
    sp_coefs1 = calc_nonlin_coefs(sp_coefs1, f_s, x_s, alpha=beta)
    refined -= sp_coefs1[:, None] * np.repeat(sp1, embs.shape[0], axis=0)
    
    
    sp_coefs2 = np.dot(refined, sp2.squeeze())
    #sp_coefs2 = beta * np.clip(sp_coefs2, -sp_ax_th, sp_ax_th)
    sp_coefs2 = calc_nonlin_coefs(sp_coefs2, f_s, x_s, alpha=beta)
    refined -= sp_coefs2[:, None] * np.repeat(sp2, embs.shape[0], axis=0)
    
    
    if filter_ood:
        
        ood_coefs1 = np.dot(refined, ood_ax1.squeeze())
        ood_coefs2 = np.dot(refined, ood_ax2.squeeze())
        refined -= ood_coefs1[:, None] * np.repeat(ood_ax1, embs.shape[0], axis=0)
        refined -= ood_coefs2[:, None] * np.repeat(ood_ax2, embs.shape[0], axis=0)
        refined += cr_coefs1[:, None] * np.repeat(cr1, embs.shape[0], axis=0)
        refined += cr_coefs2[:, None] * np.repeat(cr2, embs.shape[0], axis=0)
                
    
#    refined = normalize(refined)
    return refined


refined_grouped_embs = {}
for key in grouped_embs.keys():
    refined_grouped_embs[key] = refine_embs(grouped_embs[key], sp_ax1, sp_ax2, core_ax1, core_ax2)


refined_grouped_prototypes = {}
for key in grouped_prototypes.keys():
    refined_grouped_prototypes[key] = refine_embs(grouped_prototypes[key], sp_ax1, sp_ax2, core_ax1, core_ax2)


    
refined_ood_embs = {}
for key in ood_embs.keys():
    refined_ood_embs[key] = refine_embs(ood_embs[key], sp_ax1, sp_ax2, core_ax1, core_ax2)


refined_ood_prototypes = {group: get_prototypes(embs)\
                              for group, embs in refined_ood_embs.items()}


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



def calc_euc_dist(embs, prototypes):
    euc_dist = np.linalg.norm(embs - prototypes, axis=-1)
    return euc_dist



grouped_cos_dist = {group: calc_euc_dist(embs, refined_grouped_prototypes[group])\
                    for group, embs in refined_grouped_embs.items()}

selected_groups_names = []
for i in range(2):
  for j in range(2):
    selected_groups_names.append(f'{core_class_names[i]}_{sp_class_names[j]}')
selected_grouped_embs = {name: refined_grouped_embs[name] for name in selected_groups_names}
    
print('refined:')
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for group, ax in zip([group for group in selected_grouped_embs], axes):
    sns.histplot(grouped_cos_dist[group], label=group, palette=['red'], ax=ax, element='step', linewidth=2.5, fill=False)
    loc = group.find('_')
    sp_name = group[loc + 1:]
    ood_embs_arr = refined_ood_embs[sp_name]
    ood_dists = calc_euc_dist(ood_embs_arr, refined_grouped_prototypes[group])
    sns.histplot(ood_dists, label='ood', ax=ax, element='step', linewidth=2.5, fill=False)
    ax.legend()
    ax.set_title(group, fontsize=17)
    print(group, sp_name, np.mean(ood_dists) / np.mean(grouped_cos_dist[group]),
          wasserstein_distance(ood_dists, grouped_cos_dist[group]))
    
grouped_cos_dist = {group: calc_euc_dist(embs, grouped_prototypes[group]) for group, embs in grouped_embs.items()}

selected_grouped_embs = {name: grouped_embs[name] for name in selected_groups_names}
    
print('neutral:')
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for group, ax in zip([group for group in selected_grouped_embs], axes):
    sns.histplot(grouped_cos_dist[group], label=group, palette=['red'], ax=ax, element='step', linewidth=2.5, fill=False)
    loc = group.find('_')
    sp_name = group[loc + 1:]
    ood_embs_arr = ood_embs[sp_name]
    ood_dists = calc_euc_dist(ood_embs_arr, grouped_prototypes[group])
    sns.histplot(ood_dists, label='ood', ax=ax, element='step', linewidth=2.5, fill=False)
    ax.legend()
    ax.set_title(group, fontsize=17)
    print(group, sp_name, np.mean(ood_dists) / np.mean(grouped_cos_dist[group]),
          wasserstein_distance(ood_dists, grouped_cos_dist[group]))
    


def get_dist_vals(core_name, refined=False):
    
    if refined:
        embs = refined_grouped_embs
        p_embs = refined_grouped_prototypes
    else:
        embs = grouped_embs
        p_embs = grouped_prototypes
        
    all_dist_vals = []

    for sp in sp_class_names:
        dist_vals = [calc_euc_dist(embs[core_name + '_' + sp],
                                  p_embs[core_name + '_' + sp_name]) for sp_name in sp_class_names]
        dist_vals = np.min(dist_vals, axis=0)
        all_dist_vals.append(dist_vals)
    
    return np.concatenate(all_dist_vals)
    

def get_dist_vals_ood(core_name, refined=False):
    
    if refined:
        embs = np.concatenate([refined_ood_embs[ood_class_names[0]], refined_ood_embs[ood_class_names[1]]])
        p_embs = refined_grouped_prototypes
    else:
        embs = np.concatenate([ood_embs[ood_class_names[0]], ood_embs[ood_class_names[1]]])
        p_embs = grouped_prototypes
        
    dist_vals = [calc_euc_dist(embs,
                              p_embs[core_name + '_' + sp_name]) for sp_name in sp_class_names]
    dist_vals = np.min(dist_vals, axis=0)
    
    return dist_vals


def find_thresh_val(main_vals, th=0.95):
    thresh = np.sort(main_vals)[int(th * len(main_vals))]
    return thresh
    


for core_name in core_class_names:
    
    neutral_ood = get_dist_vals_ood(core_name)
    refined_ood = get_dist_vals_ood(core_name, refined=True)

    neutral_ind = get_dist_vals(core_name)
    refined_ind = get_dist_vals(core_name, refined=True)

            
    neutral_th = find_thresh_val(neutral_ind)
    neutral_err = neutral_ood[neutral_ood < neutral_th].shape[0] / neutral_ood.shape[0]

    
    refined_th = find_thresh_val(refined_ind)
    refined_err = refined_ood[refined_ood < refined_th].shape[0] / refined_ood.shape[0]
    
    print('neutral:', 100 * neutral_err,
          np.mean(neutral_ind) / np.mean(neutral_ood))
    
    print('refined:', 100 * refined_err,
          np.mean(np.mean(refined_ind)) / np.mean(np.mean(refined_ood)))
    
    print('***********************')
    
    
    thresholds = [th for th in np.arange(1, 100) / 100]
    
    n_fps = [0]
    n_tps = [0]
    
    r_fps = [0]
    r_tps = [0]
    
    for th in thresholds:
        
        neutral_th = find_thresh_val(neutral_ind, th)
        
        neutral_fp = neutral_ood[neutral_ood < neutral_th].shape[0] / neutral_ood.shape[0]
        neutral_tp = neutral_ind[neutral_ind < neutral_th].shape[0] / neutral_ind.shape[0]
        
        n_fps.append(neutral_fp)
        n_tps.append(neutral_tp)

        refined_th = find_thresh_val(refined_ind, th)
        
        refined_fp = refined_ood[refined_ood < refined_th].shape[0] / refined_ood.shape[0]
        refined_tp = refined_ind[refined_ind < refined_th].shape[0] / refined_ind.shape[0]
        
        r_fps.append(refined_fp)
        r_tps.append(refined_tp)
                
    n_fps.append(1)
    n_tps.append(1)
    r_fps.append(1)
    r_tps.append(1)
    
    n_auc = np.round(auc(n_fps, n_tps), 4)
    r_auc = np.round(auc(r_fps, r_tps), 4)
    
    plt.figure()
    plt.plot(n_fps, n_tps, label=f'before refinement, area={n_auc}', linewidth=2)
    plt.plot(r_fps, r_tps, label=f'after refinement, area={r_auc}', linewidth=2)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    #plt.ylim([0.55, 1.001])
    plt.legend()
    plt.title(f'ROC ({core_name})', fontsize=17)
        
