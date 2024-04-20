import numpy as np



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
    