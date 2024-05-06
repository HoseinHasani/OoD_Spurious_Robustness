import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def calc_euc_dist(embs, prototypes):
    euc_dist = np.linalg.norm(embs - prototypes, axis=-1)
    return euc_dist

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
    
    
    
def get_dist_vals(embs_dict, embs_std_dict=None, known_group=False, prototypes=None):
        
    all_dist_vals = []
    if known_group:
        for key in embs_dict.keys():
            dist_vals = [calc_euc_dist(embs_dict[key],
                                      embs_dict[k].mean(0)) for k in embs_dict.keys()]
            
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
                dist_vals = calc_euc_dist(embs_dict[key], prototype)
                
                if embs_std_dict is not None:
                    dist_vals = dist_vals * (0.001 + embs_std_dict[key])
                    
                all_dist_vals.append(dist_vals)
            
        else:
            for key in embs_dict.keys():
                p_dist_vals = []
                for p in range(len(prototypes)):
                    dist_vals = calc_euc_dist(embs_dict[key], prototypes[p])
                    
                    if embs_std_dict is not None:
                        dist_vals = dist_vals * (0.001 + embs_std_dict[key])
                    
                    p_dist_vals.append(dist_vals)
                    
                all_dist_vals.append(np.min(p_dist_vals, axis=0))
                
    return np.concatenate(all_dist_vals)
    

def get_dist_vals_ood(embs_dict, ood_embs, ood_embs_std=None, known_group=False, prototypes=None):
    
    if known_group:
        dist_vals = [calc_euc_dist(ood_embs, embs_dict[k].mean(0)) for k in embs_dict.keys()]
        
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
            
            dist_vals = calc_euc_dist(ood_embs, prototype)
            
            if ood_embs_std is not None:
                
                dist_vals = dist_vals * (0.001 + ood_embs_std)
                
        else:
            
            p_dist_vals = []
            for p in range(len(prototypes)):
                dist_vals_ = calc_euc_dist(ood_embs, prototypes[p])
                
                if ood_embs_std is not None:
                    
                    dist_vals_ = dist_vals_ * (0.001 + ood_embs_std)
                
                p_dist_vals.append(dist_vals_)
            
            dist_vals = np.min(p_dist_vals, axis=0)
            
    return dist_vals


def find_thresh_val(main_vals, th=0.95):
    thresh = np.sort(main_vals)[int(th * len(main_vals))]
    return thresh
    


def calc_ROC(embs_dict, ood_embs,
             embs_std_dict=None, ood_embs_std=None, prototypes=None,
             known_group=False, plot=False):
        
        
    ood_dists = get_dist_vals_ood(embs_dict, ood_embs, ood_embs_std, known_group, prototypes=prototypes)

    ind_dists = get_dist_vals(embs_dict, embs_std_dict, known_group, prototypes=prototypes)

            
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
    
    y = np.concatenate((np.ones(len(ind_dists)), 2*np.ones(len(ood_dists))))
    pred = np.concatenate((ind_dists, ood_dists))
    
    fps, tps, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    fps = np.concatenate([[0], fps, [1]])
    tps = np.concatenate([[0], tps, [1]])
    
    auc_val = np.round(metrics.auc(fps, tps), 4)
    
    
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
    
    
    
    print('auc: ', auc_val, ', err: ', 100 * err,
          np.mean(ind_dists) / np.mean(ood_dists))
    
    print('***********************')
    
    
    
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
    
    
    y = np.concatenate((np.ones(len(ind_dists)), 2*np.ones(len(ood_dists))))
    pred = np.concatenate((ind_dists, ood_dists))
    
    fps, tps, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    fps = np.concatenate([[0], fps, [1]])
    tps = np.concatenate([[0], tps, [1]])
    auc_val = np.round(metrics.auc(fps, tps), 4)
    
    
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
    
    
    print('auc: ', auc_val, ', err: ', 100 * err,
          np.mean(ind_dists) / np.mean(ood_dists))
    
    print('***********************')