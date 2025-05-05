from base_sprod import BaseSPROD
import numpy as np

class FullSPROD(BaseSPROD):
    def __init__(self, config, 
                 use_group_refinement: bool = False,
                 merge_refined_prototypes: bool = False,
                 refine_n_iter: int = 1):
        super().__init__(config, probabilistic_score=False, normalize_features=True)
        self.use_group_refinement = use_group_refinement
        self.merge_refined_prototypes = merge_refined_prototypes
        self.refine_n_iter = refine_n_iter
        

    def setup(self, net, id_loader_dict, ood_loader_dict):
        """Override setup to perform flexible classification-aware prototype refinement."""
        super().setup(net, id_loader_dict, ood_loader_dict)
        self.perform_classification()

    def perform_classification(self):
        """Expand, optionally refine, and optionally merge prototypes."""
        train_prototypes = np.stack(self.train_prototypes)
        x_train = np.array(self.train_feats)
        y_train = np.array(self.train_labels)

        dists = self.calc_euc_dist(x_train, train_prototypes)
        y_hat_train = np.argmin(dists, axis=1)

        total_misc_inds = np.where(y_hat_train != y_train)[0]
        total_crr_inds = np.where(y_hat_train == y_train)[0]

        final_prototypes = []

        num_classes = len(np.unique(y_train))

        for l in range(num_classes):
            group_embs = []
            class_inds = np.where(y_train == l)[0]

            # Correctly classified embeddings
            class_crr_inds = np.intersect1d(class_inds, total_crr_inds, assume_unique=True)
            if len(class_crr_inds) > 0:
                group_embs.append(x_train[class_crr_inds])

            for j in range(num_classes):
                if j == l:
                    continue
                # Misclassified to class j
                class_misc_inds = np.intersect1d(class_inds, total_misc_inds, assume_unique=True)
                trg_lbl_inds = np.where(y_hat_train == j)[0]
                class_misc_inds_trg = np.intersect1d(class_misc_inds, trg_lbl_inds, assume_unique=True)

                if len(class_misc_inds_trg) > 0:
                    group_embs.append(x_train[class_misc_inds_trg])

            # Decide what to do with the group
            if self.use_group_refinement and group_embs:
                refined_prototypes = self.refine_group_prototypes(group_embs, n_iter=self.refine_n_iter)
            else:
                refined_prototypes = [embs.mean(axis=0) for embs in group_embs]

            # SPROD4 behavior: merge
            if self.merge_refined_prototypes:
                merged_prototype = np.mean(refined_prototypes, axis=0)
                final_prototypes.append(merged_prototype)
            else:
                final_prototypes.extend(refined_prototypes)

        self.train_prototypes.extend(final_prototypes); print('extend')
        # self.train_prototypes = final_prototypes; print('equal')
        
    def refine_group_prototypes(self, group_embs, n_iter: int = 1):
        """Refine prototypes iteratively."""
        all_embs = np.concatenate(group_embs)
        prototypes = [embs.mean(axis=0) for embs in group_embs]
        prototypes = np.array(prototypes)

        for _ in range(n_iter):
            dists = self.calc_euc_dist(all_embs, prototypes)
            labels = np.argmin(dists, axis=1)
            new_embs = [all_embs[labels == l] for l in np.unique(labels)]
            prototypes = [embs.mean(axis=0) for embs in new_embs]
            prototypes = np.array(prototypes)

        return prototypes
    
    
    def numpy_inference2(self,
                        embeddings: np.ndarray,
                        labels: np.ndarray = None,
                        donormalize: bool = False):
    
        
        if donormalize:
            embeddings = self.normalize(embeddings)
            
        self.train_feats = embeddings
        self.train_labels = labels
    
        if labels is not None:
            _, _, _ = self.numpy_inference(embeddings, labels, donormalize)
            self.build_class_prototypes(embeddings, labels)
            self.setup_flag = True
            self.perform_classification()
    
        prototypes = np.stack(self.train_prototypes)
        dists = self.calc_euc_dist(embeddings, prototypes)
    
        if self.probabilistic_score:
            probs = np.exp(-dists)
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            dists = 1.0 - probs
    
        pred_list = np.argmin(dists, axis=1).astype(int)
        conf_list = -np.min(dists, axis=1)
    
        return pred_list, conf_list, labels
