from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import openood.utils.comm as comm
from .base_postprocessor import BasePostprocessor



class KMRefinedPrototypicalPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(KMRefinedPrototypicalPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.activation_log = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False
        self.perform_normalization = True
        self.train_embs = []
        self.train_prototypes = []
        self.train_labels = None
        self.train_feats = None
        
        
    def refine_group_prototypes(self, group_embs, n_iter=2):
        all_embs = np.concatenate(group_embs)
    
        prototypes = [embs.mean(0) for embs in group_embs]
        prototypes = np.array(prototypes)
        
        for k in range(n_iter):
            dists = np.linalg.norm(all_embs[..., None] - prototypes.T[None], axis=1)
            labels = np.argmin(dists, axis=1)
            new_embs = []
            for l in np.unique(labels):
                inds = np.argwhere(labels == l).ravel()
                new_embs.append(all_embs[inds])
    
            prototypes = [embs.mean(0) for embs in new_embs]
            prototypes = np.array(prototypes)
        
        
        return prototypes

    def perform_classification(self):
        
        train_prototypes = np.array(self.train_prototypes.copy())
        x_train = np.array(self.train_feats.copy())
        y_train = np.array(self.train_labels)
        dists = np.linalg.norm(x_train[..., None] - train_prototypes.T[None], axis=1)
        y_hat_train = np.argmin(dists, -1)  
        
        
        total_misc_inds = np.argwhere(y_hat_train != y_train).ravel()
        total_crr_inds = np.argwhere(y_hat_train == y_train).ravel()
        
        aug_prototypes = []

        n_c = len(np.unique(y_train))
        
        for l in range(n_c):
            class_embs = []
            class_prototypes = []
            class_inds = np.argwhere(y_train == l).ravel()
        
            class_crr_inds = np.intersect1d(class_inds, total_crr_inds, assume_unique=True)
            crr_prototype = x_train[class_crr_inds].mean(0)
            class_prototypes.append(crr_prototype)
            class_embs.append(x_train[class_crr_inds])
        
            for j in range(n_c):
                if j == l:
                    continue
                
                class_miss_inds = np.intersect1d(class_inds, total_misc_inds, assume_unique=True)
                
                if len(class_miss_inds) < 1:
                    print('*'*20)
                    print('Empty!')
                    
                trg_lbl_inds = np.argwhere(y_hat_train == j).ravel()
                class_miss_inds_trg = np.intersect1d(class_miss_inds, trg_lbl_inds, assume_unique=True)
                
                if len(class_miss_inds_trg) > 0:
                    trg_prototype = x_train[class_miss_inds_trg].mean(0)
                    class_prototypes.append(trg_prototype)
                    class_embs.append(x_train[class_miss_inds_trg])
                    
                
                
                if len(class_prototypes) > 1:
                    refined_prototypes = self.refine_group_prototypes(class_embs)
                else:
                    refined_prototypes = class_prototypes
                
            aug_prototypes.append(np.mean(refined_prototypes, 0))
            
        self.train_prototypes.extend(aug_prototypes)
        
        
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        label_list = []
        feat_list = []
        net.eval()
        with torch.no_grad():
            for batch in tqdm(id_loader_dict['train'],
                              desc='Setup: ',
                              position=0,
                              leave=True):
                feat_ = batch[0]
                if self.perform_normalization:
                    feat_ = self.normalize(feat_)
                feat_list.append(feat_)
                
                label = batch[1]
                if label.ndim == 2: # for wb
                    label = np.argmax(label, axis=-1)
                    
                label_list.append(label)
                

        features = np.concatenate(feat_list)
        labels = np.concatenate(label_list)
        self.train_labels = labels
        self.train_feats = features
        
        for c in np.unique(labels):
            inds = np.argwhere(labels == c).ravel()
            train_emb = features[inds]
            self.train_embs.append(train_emb)
            self.train_prototypes.append(train_emb.mean(0))
            #self.train_prototypes.append(np.median(train_emb, axis=0))
            
        self.perform_classification()
        self.setup_flag = True

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        pass
    
    def normalize(self, x, eps=1e-7):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
    
    def calc_euc_dist(self, embs, prototypes):
        dist = np.linalg.norm(embs - prototypes, axis=-1)
        return dist
        
    def inference(self,
                  net: nn.Module,
                  data_loaders_dict: dict,
                  progress: bool = True,
                  only_feature: bool = False,
                  model_name: str = ''
                  ):
                  
        device = next(net.parameters()).device 
        data_list, label_list = [], []
        for loader_name, data_loader in data_loaders_dict.items():
            for batch in tqdm(data_loader,
                            disable=not progress or not comm.is_main_process()):
                data = batch[0]
                label = batch[1]
                
                if self.perform_normalization:
                    data = self.normalize(data)
                    
                if label.ndim == 2: # for wb
                    label = np.argmax(label, axis=-1)
                    
                data_list.append(data)
                label_list.append(label)

        # convert values into numpy array
        
        data = np.concatenate(data_list)
        label_list = np.concatenate(label_list).astype(int)
        
        dists = np.array([self.calc_euc_dist(data, p) for p in self.train_prototypes])
        pred_list = np.argmin(dists, axis=0).astype(int)
        conf_list = -np.min(dists, axis=0)
        return pred_list, conf_list, label_list

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K
