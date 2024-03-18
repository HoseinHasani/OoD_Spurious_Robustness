import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset import GaussianDataset3D
from scipy.stats import wasserstein_distance
import seaborn as sns
from sklearn.metrics import auc

#%matplotlib qt

dataset = GaussianDataset3D(9)
g_embs = dataset.grouped_embs
ood_embs = dataset.o[0]
print(np.dot(dataset.sp_ax, dataset.core_ax))

core_class_names = ['0', '1']
sp_class_names = ['0', '1']

def draw_point_cloud3D(ax, data, label, color, alpha=0.15):
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color=color, label=label, alpha=alpha, s=2)

def plot_prototype3D(ax, data, label, color):
    ax.scatter(data[0], data[1], data[2], color=color, label=label, alpha=1, s=12)
    
def draw_arrow3D(ax, arrow, label, color, linestyle):
    ax.quiver(0, 0, 0, arrow[0], arrow[1], arrow[2],
              label=label, color=color, linestyle=linestyle, linewidth=2)
    
    
def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)
    
def refine_embs(embs, sp1, sp2, cr1, cr2, alpha=1., beta=1.):
    embs = normalize(embs)
    
    refined = 0.1 * embs.copy()
    
    cr_coefs1 = np.dot(embs, cr1.squeeze())
    refined += cr_coefs1[:, None] * np.repeat(cr1, embs.shape[0], axis=0)
    
    
#    cr_coefs2 = np.dot(embs, cr2.squeeze())
#    refined += cr_coefs2[:, None] * np.repeat(cr2, embs.shape[0], axis=0)


#    sp_coefs1 = beta * np.dot(refined, sp1.squeeze())
#    refined -= alpha * sp_coefs1[:, None] * np.repeat(sp1, embs.shape[0], axis=0)
#    
#    
#    sp_coefs2 = beta * np.dot(refined, sp2.squeeze())
#    refined -= alpha * sp_coefs2[:, None] * np.repeat(sp2, embs.shape[0], axis=0)
    
#    refined = normalize(refined)
    return refined
   
figsize = 10
fig = plt.figure(figsize=(figsize, figsize))

ax = fig.add_subplot(projection='3d')

draw_point_cloud3D(ax, g_embs['0_0'], 'maj 0', 'tab:blue')
draw_point_cloud3D(ax, g_embs['0_1'], 'min 0', 'tab:green')
draw_point_cloud3D(ax, g_embs['1_0'], 'maj 1', 'tab:orange')
draw_point_cloud3D(ax, g_embs['1_1'], 'min 1', 'tab:red')

draw_point_cloud3D(ax, ood_embs, 'OoD', 'tab:gray')

plot_prototype3D(ax, g_embs['0_0'].mean(0), 'maj 0 - prototype', 'blue')
plot_prototype3D(ax, g_embs['0_1'].mean(0), 'maj 0 - prototype', 'green')
plot_prototype3D(ax, g_embs['1_0'].mean(0), 'maj 1 - prototype', 'orange')
plot_prototype3D(ax, g_embs['1_1'].mean(0), 'maj 1 - prototype', 'red')

draw_arrow3D(ax, dataset.core_ax, 'core axis', 'red', 'solid')
draw_arrow3D(ax, dataset.sp_ax, 'spurious axis', 'orange', 'dashed')
#draw_arrow3D(ax, dataset.perp_ax, 'perp axis', 'gray', 'dashed')


fig.tight_layout()
plt.legend()


def project_points_to_plane(points_3d):
    cr_coefs1 = np.dot(dataset.sp_ax, dataset.core_ax)
    sp_ax = dataset.sp_ax - cr_coefs1 * dataset.core_ax
    normal_vector = normalize(sp_ax)
    #normal_vector = dataset.sp_ax
    projection_matrix = np.eye(3) - np.outer(normal_vector, normal_vector)
    points_2d = np.dot(points_3d, projection_matrix)
    
    return points_2d


figsize = 10
fig = plt.figure(figsize=(figsize, figsize))

ax = fig.add_subplot(projection='3d')

draw_point_cloud3D(ax, project_points_to_plane(g_embs['0_0']), 'maj 0', 'tab:blue')
draw_point_cloud3D(ax, project_points_to_plane(g_embs['0_1']), 'min 0', 'tab:green')
draw_point_cloud3D(ax, project_points_to_plane(g_embs['1_0']), 'maj 1', 'tab:orange')
draw_point_cloud3D(ax, project_points_to_plane(g_embs['1_1']), 'min 1', 'tab:red')

draw_point_cloud3D(ax, project_points_to_plane(ood_embs), 'OoD', 'tab:gray')




fig.tight_layout()
plt.legend()

def calc_cos_dist(embs, prototypes):
#    embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
#    prototypes = prototypes / np.linalg.norm(prototypes, axis=-1, keepdims=True)
    cos_dist = (1 - (embs * prototypes).sum(axis=-1)) / 2
    #cos_dist = np.abs(cos_dist)
    return cos_dist.squeeze()

#def calc_cos_dist(embs, prototypes):
#    euc_dist = np.linalg.norm(embs - prototypes, axis=-1)
#    return euc_dist

grouped_cos_dist = {group: calc_cos_dist(embs, g_embs[group].mean(0)) for group, embs in g_embs.items()}


print('neutral:')
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for group, ax in zip([group for group in g_embs], axes):
    sns.histplot(grouped_cos_dist[group], label=group, palette=['red'], ax=ax, element='step', linewidth=2.5, fill=False)
    loc = group.find('_')
    sp_name = group[loc + 1:]
    ood_embs_arr = ood_embs
    ood_dists = calc_cos_dist(ood_embs_arr, g_embs[group].mean(0))
    sns.histplot(ood_dists, label='ood', ax=ax, element='step', linewidth=2.5, fill=False)
    ax.legend()
    ax.set_title(group, fontsize=17)
    print(group, sp_name, np.mean(ood_dists) / np.mean(grouped_cos_dist[group]),
          wasserstein_distance(ood_dists, grouped_cos_dist[group]))
    
    
refined_g_embs = {}
for key in g_embs.keys():
    refined_g_embs[key] = refine_embs(g_embs[key], dataset.sp_ax[None], dataset.sp_ax[None], dataset.core_ax[None], dataset.core_ax[None])

refined_ood_embs = refine_embs(ood_embs, dataset.sp_ax[None], dataset.sp_ax[None], dataset.core_ax[None], dataset.core_ax[None])


grouped_cos_dist = {group: calc_cos_dist(embs, refined_g_embs[group].mean(0)) for group, embs in refined_g_embs.items()}


print('refined:')
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for group, ax in zip([group for group in refined_g_embs], axes):
    sns.histplot(grouped_cos_dist[group], label=group, palette=['red'], ax=ax, element='step', linewidth=2.5, fill=False)
    loc = group.find('_')
    sp_name = group[loc + 1:]
    ood_embs_arr = refined_ood_embs
    ood_dists = calc_cos_dist(ood_embs_arr, refined_g_embs[group].mean(0))
    sns.histplot(ood_dists, label='ood', ax=ax, element='step', linewidth=2.5, fill=False)
    ax.legend()
    ax.set_title(group, fontsize=17)
    print(group, sp_name, np.mean(ood_dists) / np.mean(grouped_cos_dist[group]),
          wasserstein_distance(ood_dists, grouped_cos_dist[group]))
    
    
print('***********************')
    
    
    
def get_dist_vals(emb_name1, emb_name2, pr_name1, pr_name2, refined=False):
    
    if refined:
        embs = refined_g_embs
    else:
        embs = g_embs
        
    dist_vals = calc_cos_dist(embs[emb_name1 + '_' + emb_name2],
                              embs[pr_name1 + '_' + pr_name2].mean(0))
    return dist_vals
    

def get_dist_vals_ood(pr_name1, pr_name2, refined=False):
    
    if refined:
        embs = refined_ood_embs
        p_embs = refined_g_embs
    else:
        embs = ood_embs
        p_embs = g_embs
        
    dist_vals = calc_cos_dist(embs,
                              p_embs[pr_name1 + '_' + pr_name2].mean(0))
    return dist_vals


def find_thresh_val(main_vals, th=0.95):
    thresh = np.sort(main_vals)[int(th * len(main_vals))]
    return thresh
    

    
for core_name in core_class_names:
    for sp_name in sp_class_names:
        neutral_ood = get_dist_vals_ood(core_name, sp_name)
        refined_ood = get_dist_vals_ood(core_name, sp_name, refined=True)
    
        neutral_ind = get_dist_vals(core_name, sp_name, core_name, sp_name)
        refined_ind = get_dist_vals(core_name, sp_name, core_name, sp_name, refined=True)
    
                
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
        plt.title(f'ROC ({core_name}_{sp_name})', fontsize=17)
    
