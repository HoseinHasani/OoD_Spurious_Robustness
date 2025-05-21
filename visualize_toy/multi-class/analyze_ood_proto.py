import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

m = 40
Ks = [2, 3, 4, 6, 9, 12, 18, 24] 
seeds = list(range(10, 21))
N_per_class0 = 200 
radius = 100
n_ood0 = 400

scoring_mode = "min_distance" 
# scoring_mode = "softmax" 

def generate_distant_centers(K, dim, center_std=2.0, min_dist=0.2, max_attempts=2000):
    centers = []
    attempts = 0
    while len(centers) < K and attempts < max_attempts:
        candidate = np.random.randn(dim) * center_std
        if all(np.linalg.norm(candidate - c) >= min_dist for c in centers):
            if np.linalg.norm(candidate) < 0.8 * radius:
                centers.append(candidate)
        attempts += 1
    if len(centers) < K:
        print(f"Warning: Could not find {K} well-separated centers after {max_attempts} attempts.")
        assert False
    return np.array(centers)

def compute_scores(X, prototypes, mode):
    dists = np.linalg.norm(X[:, None, :] - prototypes[None, :, :], axis=2)  # (N, K)
    if mode == "min_distance":
        return np.min(dists, axis=1)
    elif mode == "softmax":
        logits = -dists
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        max_softmax = np.max(softmax, axis=1)
        return -max_softmax  # lower confidence = more OOD-like
    else:
        raise ValueError(f"Invalid scoring_mode: {mode}")



results_dict = {}

for K in Ks:
    N_per_class = N_per_class0
    n_ood = n_ood0
    aurocs = []

    for seed in seeds:
        np.random.seed(seed)

        center_std = 2.0 + K / 60
        decay = 1 / K
        centers = generate_distant_centers(K=K, dim=m, center_std=center_std, min_dist=0.4)

        class_stds = np.random.uniform(0.1 + 0.6 * decay, 0.4 + decay, size=K)

        X_id = []
        y_id = []
        for i, (c, std) in enumerate(zip(centers, class_stds)):
            samples = np.random.randn(N_per_class, m) * std + c
            X_id.append(samples)
            y_id.extend([i] * N_per_class)
        X_id = np.vstack(X_id)
        y_id = np.array(y_id)

        X_ood = np.random.randn(n_ood, m)
        X_ood = X_ood / np.linalg.norm(X_ood, axis=1, keepdims=True)
        X_ood = X_ood * (radius + 0.5 * np.random.randn(n_ood, 1))

        prototypes = np.zeros((K, m))
        for i in range(K):
            class_samples = X_id[y_id == i]
            prototypes[i] = class_samples.mean(axis=0)

        scores_id = compute_scores(X_id, prototypes, mode=scoring_mode)
        scores_ood = compute_scores(X_ood, prototypes, mode=scoring_mode)

        labels = np.concatenate([np.zeros_like(scores_id), np.ones_like(scores_ood)])
        scores = np.concatenate([scores_id, scores_ood])
        auroc = roc_auc_score(labels, scores)
        aurocs.append(auroc)

    if aurocs:
        results_dict[K] = (np.mean(aurocs), np.std(aurocs))
    else:
        results_dict[K] = (np.nan, np.nan)


mean_aurocs = [results_dict[K][0] for K in Ks]
std_aurocs = [results_dict[K][1] for K in Ks]

plt.figure(figsize=(8, 8))
bars = plt.bar(np.arange(len(Ks)), 
               [val if not np.isnan(val) else 0 for val in mean_aurocs], 
               yerr=[err if not np.isnan(err) else 0 for err in std_aurocs],
               capsize=8, color='mediumseagreen', edgecolor='black')

for bar, val in zip(bars, mean_aurocs):
    if np.isnan(val):
        bar.set_visible(False)

plt.xlabel('Number of Classes (K)')
plt.ylabel('AUROC for OOD Detection')
plt.title(f'AUROC vs Number of Classes (K) using {scoring_mode} (m={m})')

valid_aurocs = [val for val in mean_aurocs if not np.isnan(val)]
y_min = max(0.0, min(valid_aurocs) - 0.05)
y_max = min(1.0, max(valid_aurocs) + 0.05)
plt.ylim(y_min, y_max)

plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar, mean in zip(bars, mean_aurocs):
    if not np.isnan(mean):
        plt.text(bar.get_x() + bar.get_width() / 2, mean + 0.01, f"{mean:.2f}",
                 ha='center', va='bottom', fontsize=10)

plt.xticks(ticks=np.arange(len(Ks)), labels=[str(k) for k in Ks])
plt.tight_layout()
plt.show()
