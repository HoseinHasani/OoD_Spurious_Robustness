import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

data = np.load("cifar100_resnet50_embeddings.npy")  
n_classes, n_samples, feature_dim = data.shape

Ks = [2, 3, 4, 7, 12, 20, 40, 80]
seeds = list(range(10))  
results_dict = {}

for K in Ks:
    print(f'class: {K}')
    aurocs = []

    for seed in seeds:
        np.random.seed(seed)
        
        ood_classes = list(range(10 * seed, 10 * (seed + 1)))
        all_classes = list(set(range(100)) - set(ood_classes))
        id_classes = np.random.choice(all_classes, size=K, replace=False)

        X_id = []
        y_id = []
        for i, clss in enumerate(id_classes):
            samples = data[clss][:200]  
            X_id.append(samples)
            y_id.extend([i] * len(samples))
        X_id = np.vstack(X_id)
        y_id = np.array(y_id)

        X_ood = []
        for clss in ood_classes:
            samples = data[clss][:40]  
            X_ood.append(samples)
        X_ood = np.vstack(X_ood)

        clf = LogisticRegression(solver='lbfgs', max_iter=20, multi_class='multinomial')
        clf.fit(X_id, y_id)

        probs_id = clf.predict_proba(X_id)
        msp_id = np.max(probs_id, axis=1)

        probs_ood = clf.predict_proba(X_ood)
        msp_ood = np.max(probs_ood, axis=1)

        labels = np.concatenate([np.zeros_like(msp_id), np.ones_like(msp_ood)])
        scores = np.concatenate([msp_id, msp_ood])
        auroc = roc_auc_score(labels, -scores) 
        aurocs.append(auroc)

    results_dict[K] = (np.mean(aurocs), np.std(aurocs))
mean_aurocs = [results_dict[K][0] for K in Ks]
std_aurocs = [results_dict[K][1] for K in Ks]

plt.figure(figsize=(8, 8))
bars = plt.bar(np.arange(len(Ks)),
               [val if not np.isnan(val) else 0 for val in mean_aurocs],
               yerr=[err if not np.isnan(err) else 0 for err in std_aurocs],
               capsize=8, color='cornflowerblue', edgecolor='black')

for bar, val in zip(bars, mean_aurocs):
    if np.isnan(val):
        bar.set_visible(False)

plt.xlabel('Number of Classes (K)')
plt.ylabel('AUROC for OOD Detection')
plt.title('AUROC vs Number of Classes (K) on CIFAR-100 (MSP using ResNet-50 embeddings)')

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
