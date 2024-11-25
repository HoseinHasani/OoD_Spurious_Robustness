from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import openood.utils.comm as comm
from .base_postprocessor import BasePostprocessor



class ValMJTTPrototypicalPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(ValMJTTPrototypicalPostprocessor, self).__init__(config)
        self.k_validation = 5
        self.args = self.config.postprocessor.postprocessor_args
        self.activation_log = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False
        self.perform_normalization = True
        self.train_embs = []
        self.train_prototypes = []
        self.train_labels = None
        self.train_feats = None

    def perform_classification(self):
        
        train_prototypes = np.array(self.train_prototypes.copy())
        x_train = np.array(self.train_feats.copy())
        y_train = np.array(self.train_labels)
        
        aug_prototypes = []

        n_c = len(np.unique(y_train))
        
        for l in range(n_c):
            class_inds = np.argwhere(y_train == l).ravel()
            x_train_l = x_train.copy()[class_inds]
            class_prototypes = self.val_prototypes[l].copy()
            dists = np.linalg.norm(x_train_l[..., None] - class_prototypes.T[None], axis=1)
            y_hat_train = np.argmin(dists, -1)  
            
            for ccc in np.unique(y_hat_train):
                c_inds = np.argwhere(y_hat_train == ccc).ravel()
                selected_embs = x_train_l[c_inds]
                aug_prototypes.append(selected_embs.mean(0))
                
        self.train_prototypes.extend(aug_prototypes)
        
        
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        label_list = []
        feat_list = []

        val_label_list = []
        val_feat_list = []
        val_group_list = []
        
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
                
            for batch in tqdm(id_loader_dict['val'],
                              desc='Setup: ',
                              position=0,
                              leave=True):
                feat_ = batch[0]
                if self.perform_normalization:
                    feat_ = self.normalize(feat_)
                val_feat_list.append(feat_)
                
                label = batch[1]
                if label.ndim == 2: 
                    label = np.argmax(label, axis=-1)
                val_label_list.append(label)
                val_group_list.append(batch[2])

        features = np.concatenate(feat_list)
        labels = np.concatenate(label_list)
        self.train_labels = labels
        self.train_feats = features

        self.val_groups = np.concatenate(val_group_list)
        self.val_labels = np.concatenate(val_label_list)
        self.val_feats = np.concatenate(val_feat_list)
        
        for c in np.unique(labels):
            inds = np.argwhere(labels == c).ravel()
            train_emb = features[inds]
            self.train_embs.append(train_emb)
            self.train_prototypes.append(train_emb.mean(0))
            #self.train_prototypes.append(np.median(train_emb, axis=0))


            c_inds = np.argwhere(self.val_labels == c).ravel()
            class_val_protos = []
            for g in np.unique(self.val_groups):
                g_inds = np.argwhere(self.val_labels == c).ravel()
                intersect_inds = np.intersect1d(c_inds, g_inds, assume_unique=True)
                selected_inds = np.random.choice(intersect_inds, size=self.k_validation, replace=False).ravel()
                selected_embs = self.val_feats[selected_inds]
                class_val_protos.append(selected_embs.mean(0))
            self.val_prototypes.append(class_val_protos)
            
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
