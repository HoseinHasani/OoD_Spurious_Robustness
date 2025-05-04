from typing import Any, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import openood.utils.comm as comm
import BasePostprocessor

class BaseSPROD(BasePostprocessor):
    def __init__(self, config, probabilistic_score: bool = False, normalize_features: bool = True):
        super(BaseSPROD, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.setup_flag = False

        # Configuration options
        self.probabilistic_score = probabilistic_score
        self.normalize_features = normalize_features

        # Data containers
        self.train_labels = None
        self.train_feats = None
        self.train_prototypes = []

    def normalize(self, x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Normalize input vectors to unit norm."""
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)

    def calc_euc_dist(self, embs: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances between embeddings and prototypes."""
        return np.linalg.norm(embs[:, None, :] - prototypes[None, :, :], axis=-1)

    def extract_features_and_labels(self, loader_dict: Dict[str, Any], loader_key: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """Extract (optionally normalized) features and labels from a dataloader."""
        features, labels = [], []

        for batch in tqdm(loader_dict[loader_key], desc=f'Extracting {loader_key}', position=0, leave=True):
            data, label = batch[0], batch[1]
            if self.normalize_features:
                data = self.normalize(data)
            features.append(data)
            labels.append(label)

        features = np.concatenate(features)
        labels = np.concatenate(labels).astype(int)
        return features, labels

    def build_class_prototypes(self, features: np.ndarray, labels: np.ndarray):
        """Build class prototypes from features and labels."""
        self.train_prototypes = []
        for c in np.unique(labels):
            inds = np.where(labels == c)[0]
            class_embs = features[inds]
            prototype = class_embs.mean(axis=0)
            self.train_prototypes.append(prototype)

    def setup(self, net: nn.Module, id_loader_dict: dict, ood_loader_dict: dict):
        """Setup phase: extract features and build prototypes."""
        net.eval()
        with torch.no_grad():
            features, labels = self.extract_features_and_labels(id_loader_dict, loader_key='train')

        self.train_feats = features
        self.train_labels = labels
        self.build_class_prototypes(features, labels)

        self.setup_flag = True

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, use_features = True, model_name = ''):
        """Optional post-processing after setup. Can be overridden."""
        raise NotImplementedError("postprocess() must be overridden in a subclass if needed.")

    def inference(self,
                  net: nn.Module,
                  data_loaders_dict: dict,
                  progress: bool = True,
                  use_features: bool = False,
                  model_name: str = '') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference: predict classes and confidence scores."""
        net.eval()
        with torch.no_grad():
            data_list, label_list = [], []

            for loader_name, data_loader in data_loaders_dict.items():
                for batch in tqdm(data_loader, disable=not progress or not comm.is_main_process()):
                    data, label = batch[0], batch[1]
                    if self.normalize_features:
                        data = self.normalize(data)
                    data_list.append(data)
                    label_list.append(label)

            data = np.concatenate(data_list)
            labels = np.concatenate(label_list).astype(int)

        prototypes = np.stack(self.train_prototypes)
        dists = self.calc_euc_dist(data, prototypes)

        if self.probabilistic_score:
            probs = np.exp(-dists)
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            dists = 1.0 - probs

        pred_list = np.argmin(dists, axis=1).astype(int)
        conf_list = -np.min(dists, axis=1)

        return pred_list, conf_list, labels
