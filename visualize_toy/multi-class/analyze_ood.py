import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

m = 40  
Ks = [2, 3, 4, 6, 9, 12, 16, 20, 25, 30]  
seeds = list(range(10, 21))
N_per_class = 200
radius = 100
n_ood = 500

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

results = []

for K in Ks:
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

        clf = LogisticRegression(solver='lbfgs', max_iter=200, multi_class='multinomial')
        clf.fit(X_id, y_id)

        probs_id = clf.predict_proba(X_id)
        msp_id = np.max(probs_id, axis=1)
        probs_ood = clf.predict_proba(X_ood)
        msp_ood = np.max(probs_ood, axis=1)

        labels = np.concatenate([np.zeros_like(msp_id), np.ones_like(msp_ood)])
        scores = np.concatenate([msp_id, msp_ood])
        auroc = roc_auc_score(labels, -scores)
        aurocs.append(auroc)

    results.append((K, np.mean(aurocs), np.std(aurocs)))

Ks_vals, mean_aurocs, std_aurocs = zip(*results)

plt.figure(figsize=(10, 6))
bars = plt.bar(Ks_vals, mean_aurocs, yerr=std_aurocs, capsize=8, color='skyblue', edgecolor='black')

plt.xlabel('Number of Classes (K)')
plt.ylabel('AUROC for OOD Detection')
plt.title(f'AUROC vs Number of Classes (K) using MSP (m={m})')
plt.ylim(0.3, 0.8)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar, mean in zip(bars, mean_aurocs):
    plt.text(bar.get_x() + bar.get_width()/2, mean + 0.02, f"{mean:.2f}", 
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
