from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import openood.utils.comm as comm
from .base_postprocessor import BasePostprocessor



class PrototypicalPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(PrototypicalPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.activation_log = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False
        self.perform_normalization = False
        self.train_embs = []
        self.train_prototypes = []
        self.train_labels = None
        self.train_feats = None

        
        
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
            # self.train_prototypes.append(np.median(train_emb, axis=0))
            
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

    # def set_hyperparam(self, hyperparam: list):
    #     self.K = hyperparam[0]

    # def get_hyperparam(self):
    #     return self.K
