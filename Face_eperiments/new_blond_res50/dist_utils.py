import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis


def calculate_mahalanobis_distance(sample, prototype, inv_cov_matrix):
    distance = mahalanobis(sample, prototype, inv_cov_matrix)
    return distance


def calc_euc_dist(embs, prototypes, cov=None):
    if cov is None:
        # dist = 1 - embs @ prototypes
        dist = np.linalg.norm(embs - prototypes, axis=-1)
    else:
        inv_cov_matrix = np.linalg.inv(cov)
        dist = np.zeros(len(embs))
        for i in range(len(embs)):
            dist[i] = calculate_mahalanobis_distance(embs[i], prototypes, inv_cov_matrix)
        
        nan_inds = np.argwhere(np.isnan(dist)).ravel()
        max_val = dist[np.argwhere(~np.isnan(dist)).ravel()].max()
        dist[nan_inds] = max_val
        
    return dist

def calc_dists_ratio(ind_dict, ood_dict):
    
    #protos = [ind_dict[name].mean(0) for name in ind_dict.keys()]
    
    ind_dists = [np.linalg.norm(ind_dict[name] - ind_dict[name].mean(0)) for name in ind_dict.keys()]
    ood_dists = []
    for ooo in ood_dict.keys():
        ood_dists_ = [np.linalg.norm(ood_dict[ooo] - ind_dict[name].mean(0)) for name in ind_dict.keys()]
        ood_dists.append(ood_dists_)
    
    ind_dists = np.array(ind_dists)
    ood_dists = np.array(ood_dists)
    
    ratio = ood_dists / ind_dists
    ratio = ratio.mean()
    ratio = np.round(ratio, 4)
    
    print('ratio:', ratio)
    
    
    
def get_dist_vals(embs_dict, embs_std_dict=None, known_group=False,
                  prototypes=None, cov=None, apply_softmax=False):
        
    all_dist_vals = []
    all_group_dists = []
    
    if known_group:
        for key in embs_dict.keys():
            dist_vals = [calc_euc_dist(embs_dict[key],
                                      embs_dict[k].mean(0),
                                      cov) for k in embs_dict.keys()]
            
            if embs_std_dict is not None:
                new_dist_vals = []
                for j in range(len(embs_dict.keys())):
                    new_dist_vals.append(dist_vals[j] * (0.001 + embs_std_dict[key]))
                    
                dist_vals = new_dist_vals
                
            dist_vals = np.min(dist_vals, axis=0)
            all_dist_vals.append(dist_vals)
    
    else:
        
        if prototypes is None:
            embs_dict_all = np.concatenate([embs_dict[k] for k in embs_dict.keys()])
            prototype = embs_dict_all.mean(0)
            
            for key in embs_dict.keys():
                dist_vals = calc_euc_dist(embs_dict[key], prototype, cov)
                
                if embs_std_dict is not None:
                    dist_vals = dist_vals * (0.001 + embs_std_dict[key])
                    
                all_dist_vals.append(dist_vals)
            
        else:
            for key in embs_dict.keys():
                p_dist_vals = []
                for p in range(len(prototypes)):
                    
                    if cov is None:
                        cov_ = cov
                    else:
                        if cov.ndim == 2:
                            cov_ = cov.copy()
                        else:
                            cov_ = cov[p].copy()
                        
                    dist_vals = calc_euc_dist(embs_dict[key], prototypes[p], cov_)
                    
                    if embs_std_dict is not None:
                        dist_vals = dist_vals * (0.001 + embs_std_dict[key])
                    
                    p_dist_vals.append(dist_vals)
                    
                if apply_softmax:
                    p_dist_vals = np.array(p_dist_vals)
                    p_dist_probs = np.exp(-p_dist_vals)
                    p_dist_probs = p_dist_probs / p_dist_probs.sum(0)
                    
                    p_dist_vals = 1 - p_dist_probs
                    
                all_dist_vals.append(np.min(p_dist_vals, axis=0))
                all_group_dists.append(np.array(p_dist_vals).T)
                
    return np.concatenate(all_dist_vals)#, np.concatenate(all_group_dists)
    

def get_dist_vals_ood(embs_dict, ood_embs, ood_embs_std=None,
                      known_group=False, prototypes=None, cov=None, apply_softmax=False):
    
    if known_group:
        dist_vals = [calc_euc_dist(ood_embs, embs_dict[k].mean(0), cov) for k in embs_dict.keys()]
        
        if ood_embs_std is not None:
            new_dist_vals = []
            for j in range(len(embs_dict.keys())):
                new_dist_vals.append(dist_vals[j] * (0.001 + ood_embs_std))
            
            dist_vals = new_dist_vals
                
        dist_vals = np.min(dist_vals, axis=0)
    
    else:
        
        if prototypes is None:
            embs_dict_all = np.concatenate([embs_dict[k] for k in embs_dict.keys()])
            prototype = embs_dict_all.mean(0)
            
            dist_vals = calc_euc_dist(ood_embs, prototype, cov)
            
            if ood_embs_std is not None:
                
                dist_vals = dist_vals * (0.001 + ood_embs_std)
                
        else:
            
            p_dist_vals = []
            for p in range(len(prototypes)):
                
                if cov is None:
                    cov_ = cov
                else:
                    if cov.ndim == 2:
                        cov_ = cov.copy()
                    else:
                        cov_ = cov[p].copy()
                        
                dist_vals_ = calc_euc_dist(ood_embs, prototypes[p], cov_)
                
                if ood_embs_std is not None:
                    
                    dist_vals_ = dist_vals_ * (0.001 + ood_embs_std)
                
                p_dist_vals.append(dist_vals_)
            
            if apply_softmax:
                p_dist_vals = np.array(p_dist_vals)
                p_dist_probs = np.exp(-p_dist_vals)
                p_dist_probs = p_dist_probs / p_dist_probs.sum(0)
                
                p_dist_vals = 1 - p_dist_probs
                
            dist_vals = np.min(p_dist_vals, axis=0)
            
    return dist_vals#, np.array(p_dist_vals).T


def find_thresh_val(main_vals, th=0.95):
    thresh = np.sort(main_vals)[int(th * len(main_vals))]
    return thresh
    


def calc_ROC(embs_dict, ood_embs,
             embs_std_dict=None, ood_embs_std=None, prototypes=None,
             known_group=False, plot=False, cov=None,
             exp_name='', network_name='', apply_exp=False, apply_softmax=False):
        
        
    ood_dists = get_dist_vals_ood(embs_dict, ood_embs, ood_embs_std, known_group, prototypes=prototypes, cov=cov, apply_softmax=apply_softmax)
    
    ind_dists = get_dist_vals(embs_dict, embs_std_dict, known_group, prototypes=prototypes, cov=cov, apply_softmax=apply_softmax)
    
    if apply_exp:
        ood_dists = -np.exp(-ood_dists)
        ind_dists = - np.exp(-ind_dists)
        
    # ind_embs = np.concatenate([embs_dict[key] for key in embs_dict.keys()])
    # ood_dists = np.concatenate([np.linalg.norm(ood_embs - prototypes[k][None], axis=-1) for k in range(len(prototypes))])
    # ind_dists = np.concatenate([np.linalg.norm(ind_embs - prototypes[k][None], axis=-1) for k in range(len(prototypes))])
            
    thresh = find_thresh_val(ind_dists)
    err = ood_dists[ood_dists < thresh].shape[0] / ood_dists.shape[0]
    
    
    # thresholds = [th for th in np.arange(1, 100) / 100]
    
    # fps = [0]
    # tps = [0]
    
    
    # for th in thresholds:
        
    #     thresh = find_thresh_val(ind_dists)
        
    #     fp = ood_dists[ood_dists < thresh].shape[0] / ood_dists.shape[0]
    #     tp = ind_dists[ind_dists < thresh].shape[0] / ind_dists.shape[0]
        
    #     fps.append(fp)
    #     tps.append(tp)
                
    # fps.append(1)
    # tps.append(1)
    
    # auc_val = np.round(metrics.auc(fps, tps), 4)
    
    y = np.concatenate((np.zeros_like(ood_dists), np.ones_like(ind_dists)))
    pred = np.concatenate((ood_dists, ind_dists))
    pred = pred.max() - pred
    
    fps, tps, thresholds = metrics.roc_curve(y, pred)
    # fps = np.concatenate([[0], fps, [1]])
    # tps = np.concatenate([[0], tps, [1]])
    
    auc_val = np.round(metrics.auc(fps, tps), 4)
    
    precs, recs, thresholds = metrics.precision_recall_curve(y, pred)
    # precs = np.concatenate([[0], precs, [1]])
    # recs = np.concatenate([[1], recs, [0]])
    
    aupr_val = np.round(metrics.auc(np.sort(precs), recs), 4)
    
    # print('****** metrics 1 ******')
    # print('auc: ', auc_val, 'aupr:', aupr_val)
    # print('****** metrics 2 ******')
    print('auc: ', np.round(metrics.roc_auc_score(y, pred), 4),
          'aupr:', np.round(metrics.average_precision_score(y, pred), 4))
    
    if plot:
        plt.figure()
        plt.plot(fps, tps, label=f'area={auc_val}', linewidth=2)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        #plt.ylim([0.55, 1.001])
        plt.legend()
        plt.title('ROC', fontsize=17)
        
        name = f'All Distances - {exp_name} - {network_name}'
        plt.figure(figsize=(6, 3))
        plt.hist(ind_dists, 25, histtype='step', density=False, linewidth=2., label='InD distances', color='tab:blue')
        plt.hist(ood_dists, 25, histtype='step', density=False, linewidth=2., label='OoD distances', color='tab:orange')
        plt.title(name)
        plt.legend()
        plt.savefig(name + '.png', dpi=130)
        
    
            
    print('95-percent err: ', np.round(100 * err, 3))
    
    # print('***********************')
    print()
    
    
    
def calc_ROC_classwise(embs_dict, ood_embs,
             embs_std_dict=None, ood_embs_std=None, prototypes=None,
             known_group=False, plot=False, cov=None,
             exp_name='', network_name=''):
        
        
    ood_dists, group_ood_dists = get_dist_vals_ood(embs_dict, ood_embs, ood_embs_std, known_group, prototypes=prototypes, cov=cov)
    
    ind_dists, group_ind_dists = get_dist_vals(embs_dict, embs_std_dict, known_group, prototypes=prototypes, cov=cov)

            
    thresh = find_thresh_val(ind_dists)
    err = ood_dists[ood_dists < thresh].shape[0] / ood_dists.shape[0]
    
    
    
    y = np.concatenate((np.zeros_like(ood_dists), np.ones_like(ind_dists)))
    pred = np.concatenate((ood_dists, ind_dists))
    pred = pred.max() - pred
    
    fps, tps, thresholds = metrics.roc_curve(y, pred)
    
    auc_val = np.round(metrics.auc(fps, tps), 4)
    
    precs, recs, thresholds = metrics.precision_recall_curve(y, pred)
    
    aupr_val = np.round(metrics.auc(np.sort(precs), recs), 4)
    
    print('auc: ', np.round(metrics.roc_auc_score(y, pred), 4),
          'aupr:', np.round(metrics.average_precision_score(y, pred), 4))
    
    if plot:
        plt.figure()
        plt.plot(fps, tps, label=f'area={auc_val}', linewidth=2)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        #plt.ylim([0.55, 1.001])
        plt.legend()
        plt.title('ROC', fontsize=17)
        
        name = f'All Distances - {exp_name} - {network_name}'
        plt.figure(figsize=(6, 3))
        plt.hist(ind_dists, 25, histtype='step', density=False, linewidth=2., label='InD distances', color='tab:blue')
        plt.hist(ood_dists, 25, histtype='step', density=False, linewidth=2., label='OoD distances', color='tab:orange')
        plt.title(name)
        plt.legend()
        plt.savefig(name + '.png', dpi=130)
        
        if len(prototypes) == 2:
            plt.figure(figsize=(9, 3))
            name = f'Class Distances - {exp_name} - {network_name}'
            plt.suptitle(name)   
            
            plt.subplot(121)
            plt.hist(group_ind_dists[:, 0], 25, histtype='step', density=False, linewidth=2., label='InD distances', color='tab:blue')
            plt.hist(group_ood_dists[:, 0], 25, histtype='step', density=False, linewidth=2., label='OoD distances', color='tab:orange')
            plt.title('class 0', y=-0.1)
            
            plt.subplot(122)
            plt.hist(group_ind_dists[:, 1], 25, histtype='step', density=False, linewidth=2., label='InD distances', color='tab:blue')
            plt.hist(group_ood_dists[:, 1], 25, histtype='step', density=False, linewidth=2., label='OoD distances', color='tab:orange')
            plt.title('class 1', y=-0.1)
            
            plt.legend()
            plt.savefig(name + '.png', dpi=130)
    
    
        if len(prototypes) == 4:
            plt.figure(figsize=(9, 6))
            name = f'Class Distances - {exp_name} - {network_name}'
            plt.suptitle(name)   
            
            plt.subplot(221)
            plt.hist(group_ind_dists[:, 0], 25, histtype='step', density=False, linewidth=2., label='InD distances', color='tab:blue')
            plt.hist(group_ood_dists[:, 0], 25, histtype='step', density=False, linewidth=2., label='OoD distances', color='tab:orange')
            plt.title('group 0', y=-0.1)

            plt.subplot(222)
            plt.hist(group_ind_dists[:, 1], 25, histtype='step', density=False, linewidth=2., label='InD distances', color='tab:blue')
            plt.hist(group_ood_dists[:, 1], 25, histtype='step', density=False, linewidth=2., label='OoD distances', color='tab:orange')
            plt.title('group 1', y=-0.1)
            
            plt.subplot(223)
            plt.hist(group_ind_dists[:, 2], 25, histtype='step', density=False, linewidth=2., label='InD distances', color='tab:blue')
            plt.hist(group_ood_dists[:, 2], 25, histtype='step', density=False, linewidth=2., label='OoD distances', color='tab:orange')
            plt.title('group 2', y=-0.1)

            plt.subplot(224)
            plt.hist(group_ind_dists[:, 3], 25, histtype='step', density=False, linewidth=2., label='InD distances', color='tab:blue')
            plt.hist(group_ood_dists[:, 3], 25, histtype='step', density=False, linewidth=2., label='OoD distances', color='tab:orange')
            plt.title('group 3', y=-0.1)
            
            plt.legend()
            plt.savefig(name + '.png', dpi=130)
            
    print('95-percent err: ', np.round(100 * err, 3))
    
    # print('***********************')
    print()
    
def calc_ROC_with_dists(ind_dists, ood_dists, plot=False):
        
        

            
    thresh = find_thresh_val(ind_dists)
    err = ood_dists[ood_dists < thresh].shape[0] / ood_dists.shape[0]
    
    
    
    y = np.concatenate((np.zeros_like(ood_dists), np.ones_like(ind_dists)))
    pred = np.concatenate((ood_dists, ind_dists))
    pred = pred.max() - pred
    
    fps, tps, thresholds = metrics.roc_curve(y, pred)
    fps = np.concatenate([[0], fps, [1]])
    tps = np.concatenate([[0], tps, [1]])
    
    auc_val = np.round(metrics.auc(fps, tps), 4)
    
    precs, recs, thresholds = metrics.precision_recall_curve(y, pred)
    precs = np.concatenate([[0], precs, [1]])
    recs = np.concatenate([[1], recs, [0]])
    
    aupr_val = np.round(metrics.auc(np.sort(precs), recs), 4)
    
    # print('****** metrics 1 ******')
    # print('auc: ', auc_val, 'aupr:', aupr_val)
    # print('****** metrics 2 ******')
    print('auc: ', np.round(metrics.roc_auc_score(y, pred), 4),
          'aupr:', np.round(metrics.average_precision_score(y, pred), 4))
    
    if plot:
        plt.figure()
        plt.plot(fps, tps, label=f'area={auc_val}', linewidth=2)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        #plt.ylim([0.55, 1.001])
        plt.legend()
        plt.title('ROC', fontsize=17)
        
        
        plt.figure(figsize=(8,4))
        plt.hist(ind_dists, 25, histtype='step', density=False, linewidth=2.5, label='InD distances', color='tab:blue')
        plt.hist(ood_dists, 25, histtype='step', density=False, linewidth=2.5, label='OoD distances', color='tab:orange')
        plt.title('dist hist')
        plt.legend()
    
    
    
    print('95-percent err: ', np.round(100 * err, 3))
    
    # print('***********************')
    print()
    
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x))
    summation = e_x.sum(axis=axis)
    probs = e_x / np.expand_dims(summation, axis)
    return probs

def calc_probs_ROC(log_dict, ood_logits, plot=False):
        
        
    ood_dists = 1 - np.max(softmax(ood_logits), -1)

    ind_dists = [np.max(softmax(log_dict[key]), -1) for key in log_dict.keys()]
    ind_dists = 1 - np.concatenate(ind_dists)
            
    thresh = find_thresh_val(ind_dists)
    err = ood_dists[ood_dists < thresh].shape[0] / ood_dists.shape[0]
    
    
    y = np.concatenate((np.zeros_like(ind_dists), np.ones_like(ood_dists)))
    pred = np.concatenate((ind_dists, ood_dists))
    
    fps, tps, thresholds = metrics.roc_curve(y, pred)
    fps = np.concatenate([[0], fps, [1]])
    tps = np.concatenate([[0], tps, [1]])
    
    auc_val = np.round(metrics.auc(fps, tps), 5)
    
    precs, recs, thresholds = metrics.precision_recall_curve(y, pred)
    precs = np.concatenate([[0], precs, [1]])
    recs = np.concatenate([[1], recs, [0]])
    
    aupr_val = np.round(metrics.auc(np.sort(precs), recs), 5)
    
    print('****** metrics 1 ******')
    print('auc: ', auc_val, 'aupr:', aupr_val)
    print('****** metrics 2 ******')
    print('auc: ', np.round(metrics.roc_auc_score(y, pred), 5),
          'aupr:', np.round(metrics.average_precision_score(y, pred), 5))
    
    if plot:
        plt.figure()
        plt.plot(fps, tps, label=f'area={auc_val}', linewidth=2)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        #plt.ylim([0.55, 1.001])
        plt.legend()
        plt.title('ROC', fontsize=17)
        
        
        plt.figure(figsize=(8,4))
        plt.hist(ind_dists, 25, histtype='step', density=False, linewidth=2.5, label='InD distances', color='tab:blue')
        plt.hist(ood_dists, 25, histtype='step', density=False, linewidth=2.5, label='OoD distances', color='tab:orange')
        plt.title('dist hist')
        plt.legend()
    
    
    
    print('95-percent err: ', np.round(100 * err, 3),
          np.mean(ind_dists) / np.mean(ood_dists))
    
    print('***********************')
    print()