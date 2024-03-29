import numpy as np
from matplotlib import pyplot as plt
from dataset import GaussianDataset2D
from scipy.stats import wasserstein_distance
import seaborn as sns
from sklearn.metrics import auc
from scipy import integrate

#%matplotlib qt

dataset = GaussianDataset2D(2, normal=False)
g_embs = dataset.grouped_embs
ood_embs = dataset.o[0]

core_ax_th = 0.4
sp_ax_th = 0.8

print(np.dot(dataset.sp_ax, dataset.core_ax))

core_class_names = ['0', '1']
sp_class_names = ['0', '1']

def draw_point_cloud3D(ax, data, label, color, alpha=0.15):
    ax.scatter(data[:, 0], data[:, 1], color=color, label=label, alpha=alpha, s=2)

def plot_prototype3D(ax, data, label, color):
    ax.scatter(data[0], data[1], color=color, label=label, marker='*', alpha=1, s=65)
    
def draw_arrow3D(ax, arrow, label, color, linestyle):
    ax.quiver(0, 0, arrow[0], arrow[1],
              label=label, color=color, linestyle=linestyle, linewidth=2)
    
    
def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def calc_stats(data, cr_ax, sp_ax):
    
    cr_coefs = []
    sp_coefs = []
    for key in data.keys():
        cr_coefs.append(np.abs(np.dot(data[key], cr_ax)))
        sp_coefs.append(np.abs(np.dot(data[key], sp_ax)))
    
    return np.concatenate(cr_coefs), np.concatenate(sp_coefs)

def calc_CDF(data):
    h, x = np.histogram(data, bins=1000, normed=True)
    F = np.cumsum(h) * (x[1] - x[0])
    return F, x[1:]


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def calc_nonlin_coefs(coefs, F, x, alpha=1):
    f = integrate.cumtrapz(1 - F ** (0.5))
    f = f / f.max() * alpha
    coef_vals = []
    for coef in coefs:
        
        if np.abs(coef) > x.max():
            coef_vals.append(alpha * coef / np.abs(coef))
        else:
            ind = find_nearest(x[:-1], np.abs(coef))
            coef_vals.append(f[ind] * coef / np.abs(coef) * alpha)
            
    return np.array(coef_vals)

    
c_vals, s_vals = calc_stats(g_embs, dataset.core_ax, dataset.sp_ax)
f_c, x_c = calc_CDF(c_vals)
f_s, x_s = calc_CDF(s_vals)

c_vals_ood, s_vals_ood = calc_stats({'o': ood_embs}, dataset.core_ax, dataset.sp_ax)
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


def refine_embs(embs, sp1, sp2, cr1, cr2, alpha=4., beta=0.5):
    
    refined = 1.0 * embs.copy()

#    cr_coefs1 = np.dot(dataset.sp_ax, dataset.core_ax)
#    core_ax = dataset.core_ax - cr_coefs1 * dataset.sp_ax
#    cr1 = normalize(core_ax)[None]
    
    cr_coefs1 = np.dot(embs, cr1.squeeze())
    #cr_coefs1 = alpha * np.clip(cr_coefs1, -core_ax_th, core_ax_th)
    cr_coefs1 = calc_nonlin_coefs(cr_coefs1, f_c, x_c, alpha=alpha)
    refined += cr_coefs1[:, None] * np.repeat(cr1, embs.shape[0], axis=0)
    
    

#    cr_coefs1 = np.dot(dataset.sp_ax, dataset.core_ax)
#    sp_ax = dataset.sp_ax - cr_coefs1 * dataset.core_ax
#    sp1 = normalize(sp_ax)[None]
    
    sp_coefs1 = np.dot(refined, sp1.squeeze())
    #sp_coefs1 = beta * np.clip(sp_coefs1, -sp_ax_th, sp_ax_th)
    sp_coefs1 = calc_nonlin_coefs(sp_coefs1, f_s, x_s, alpha=beta)
    final_refined = refined - sp_coefs1[:, None] * np.repeat(sp1, embs.shape[0], axis=0)
    signs = np.sign(final_refined / refined)
    signs[signs < 1] = 0.
    final_refined = final_refined * signs
    
#    final_refined = normalize(final_refined)
    return final_refined
   
    
   
figsize = 10
fig = plt.figure(figsize=(figsize, figsize))

ax = fig.add_subplot()

draw_point_cloud3D(ax, g_embs['0_0'], 'maj 0', 'tab:blue')
draw_point_cloud3D(ax, g_embs['0_1'], 'min 0', 'tab:green')
draw_point_cloud3D(ax, g_embs['1_0'], 'min 1', 'tab:orange')
draw_point_cloud3D(ax, g_embs['1_1'], 'maj 1', 'tab:red')

draw_point_cloud3D(ax, ood_embs, 'OoD', 'tab:gray')

plot_prototype3D(ax, g_embs['0_0'].mean(0), 'maj 0 - prototype', 'blue')
plot_prototype3D(ax, g_embs['0_1'].mean(0), 'min 0 - prototype', 'green')
plot_prototype3D(ax, g_embs['1_0'].mean(0), 'min 1 - prototype', 'orange')
plot_prototype3D(ax, g_embs['1_1'].mean(0), 'maj 1 - prototype', 'red')

draw_arrow3D(ax, dataset.core_ax, 'core axis', 'red', 'solid')
draw_arrow3D(ax, dataset.sp_ax, 'spurious axis', 'orange', 'dashed')
#draw_arrow3D(ax, dataset.perp_ax, 'perp axis', 'gray', 'dashed')


fig.tight_layout()
plt.legend()



refined_g_embs = {}
for key in g_embs.keys():
    refined_g_embs[key] = refine_embs(g_embs[key], dataset.sp_ax[None], dataset.sp_ax[None], dataset.core_ax[None], dataset.core_ax[None])

refined_ood_embs = refine_embs(ood_embs, dataset.sp_ax[None], dataset.sp_ax[None], dataset.core_ax[None], dataset.core_ax[None])


fig = plt.figure(figsize=(figsize, figsize))

ax = fig.add_subplot()

draw_point_cloud3D(ax, refined_g_embs['0_0'], 'maj 0', 'tab:blue')
draw_point_cloud3D(ax, refined_g_embs['0_1'], 'min 0', 'tab:green')
draw_point_cloud3D(ax, refined_g_embs['1_0'], 'min 1', 'tab:orange')
draw_point_cloud3D(ax, refined_g_embs['1_1'], 'maj 1', 'tab:red')

draw_point_cloud3D(ax, refined_ood_embs, 'OoD', 'tab:gray')

plot_prototype3D(ax, refined_g_embs['0_0'].mean(0), 'maj 0 - prototype', 'blue')
plot_prototype3D(ax, refined_g_embs['0_1'].mean(0), 'min 0 - prototype', 'green')
plot_prototype3D(ax, refined_g_embs['1_0'].mean(0), 'min 1 - prototype', 'orange')
plot_prototype3D(ax, refined_g_embs['1_1'].mean(0), 'maj 1 - prototype', 'red')

draw_arrow3D(ax, dataset.core_ax, 'core axis', 'red', 'solid')
draw_arrow3D(ax, dataset.sp_ax, 'spurious axis', 'orange', 'dashed')


fig.tight_layout()
plt.legend()

#def calc_cos_dist(embs, prototypes):
##    embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
##    prototypes = prototypes / np.linalg.norm(prototypes, axis=-1, keepdims=True)
#    cos_dist = (1 - (embs * prototypes).sum(axis=-1)) / 2
#    #cos_dist = np.abs(cos_dist)
#    return cos_dist.squeeze()

def calc_euc_dist(embs, prototypes):
    euc_dist = np.linalg.norm(embs - prototypes, axis=-1)
    return euc_dist

grouped_cos_dist = {group: calc_euc_dist(embs, g_embs[group].mean(0)) for group, embs in g_embs.items()}


print('neutral:')
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for group, ax in zip([group for group in g_embs], axes):
    sns.histplot(grouped_cos_dist[group], label=group, palette=['red'], ax=ax, element='step', linewidth=2.5, fill=False)
    loc = group.find('_')
    sp_name = group[loc + 1:]
    ood_embs_arr = ood_embs
    ood_dists = calc_euc_dist(ood_embs_arr, g_embs[group].mean(0))
    sns.histplot(ood_dists, label='ood', ax=ax, element='step', linewidth=2.5, fill=False)
    ax.legend()
    ax.set_title(group, fontsize=17)
    print(group, sp_name, np.mean(ood_dists) / np.mean(grouped_cos_dist[group]),
          wasserstein_distance(ood_dists, grouped_cos_dist[group]))
    
    

grouped_cos_dist = {group: calc_euc_dist(embs, refined_g_embs[group].mean(0)) for group, embs in refined_g_embs.items()}


print('refined:')
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for group, ax in zip([group for group in refined_g_embs], axes):
    sns.histplot(grouped_cos_dist[group], label=group, palette=['red'], ax=ax, element='step', linewidth=2.5, fill=False)
    loc = group.find('_')
    sp_name = group[loc + 1:]
    ood_embs_arr = refined_ood_embs
    ood_dists = calc_euc_dist(ood_embs_arr, refined_g_embs[group].mean(0))
    sns.histplot(ood_dists, label='ood', ax=ax, element='step', linewidth=2.5, fill=False)
    ax.legend()
    ax.set_title(group, fontsize=17)
    print(group, sp_name, np.mean(ood_dists) / np.mean(grouped_cos_dist[group]),
          wasserstein_distance(ood_dists, grouped_cos_dist[group]))
    
    
print('***********************')
    
    
    
def get_dist_vals(core_name, refined=False):
    
    if refined:
        embs = refined_g_embs
    else:
        embs = g_embs
        
    all_dist_vals = []

    for sp in sp_class_names:
        dist_vals = [calc_euc_dist(embs[core_name + '_' + sp],
                                  embs[core_name + '_' + sp_name].mean(0)) for sp_name in sp_class_names]
        dist_vals = np.min(dist_vals, axis=0)
        all_dist_vals.append(dist_vals)
    
    return np.concatenate(all_dist_vals)
    

def get_dist_vals_ood(core_name, refined=False):
    
    if refined:
        embs = refined_ood_embs
        p_embs = refined_g_embs
    else:
        embs = ood_embs
        p_embs = g_embs
        
    dist_vals = [calc_euc_dist(embs,
                              p_embs[core_name + '_' + sp_name].mean(0)) for sp_name in sp_class_names]
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
