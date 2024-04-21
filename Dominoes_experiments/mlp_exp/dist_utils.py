import numpy as np
from sklearn.metrics import auc
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
    
    
    
def get_dist_vals(embs_dict):
    
        
    all_dist_vals = []

    for key in embs_dict.keys():
        dist_vals = [calc_euc_dist(embs_dict[key],
                                  embs_dict[k].mean(0)) for k in embs_dict.keys()]
        dist_vals = np.min(dist_vals, axis=0)
        all_dist_vals.append(dist_vals)
    
    return np.concatenate(all_dist_vals)
    

def get_dist_vals_ood(embs_dict, ood_embs):
    
    dist_vals = [calc_euc_dist(ood_embs, embs_dict[k].mean(0)) for k in embs_dict.keys()]
    dist_vals = np.min(dist_vals, axis=0)
    
    return dist_vals


def find_thresh_val(main_vals, th=0.95):
    thresh = np.sort(main_vals)[int(th * len(main_vals))]
    return thresh
    


def calc_ROC(embs_dict, ood_embs, plot=False):
        
    ood_dists = get_dist_vals_ood(embs_dict, ood_embs)

    ind_dists = get_dist_vals(embs_dict)

            
    thresh = find_thresh_val(ind_dists)
    err = ood_dists[ood_dists < thresh].shape[0] / ood_dists.shape[0]

    
    

    
    
    thresholds = [th for th in np.arange(1, 100) / 100]
    
    fps = [0]
    tps = [0]
    
    
    for th in thresholds:
        
        thresh = find_thresh_val(ind_dists)
        
        fp = ood_dists[ood_dists < thresh].shape[0] / ood_dists.shape[0]
        tp = ind_dists[ind_dists < thresh].shape[0] / ind_dists.shape[0]
        
        fps.append(fp)
        tps.append(tp)
                
    fps.append(1)
    tps.append(1)
    
    auc_val = np.round(auc(fps, tps), 4)
    
    if plot:
        plt.figure()
        plt.plot(fps, tps, label=f'area={auc_val}', linewidth=2)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        #plt.ylim([0.55, 1.001])
        plt.legend()
        plt.title('ROC', fontsize=17)
    
    
    print('auc: ', auc_val, ', err: ', 100 * err,
          np.mean(ind_dists) / np.mean(ood_dists))
    
    print('***********************')