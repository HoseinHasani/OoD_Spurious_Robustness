import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

id_data = np.load("SPI_ID_resnet50.npy")  # (100, N, 2048)
ood_data = np.load("SPI_OOD_resnet50.npy")  # shape: (100, N, 2048)

id_data = id_data / np.linalg.norm(id_data, 2, axis=-1, keepdims=True)
ood_data = ood_data / np.linalg.norm(ood_data, 2, axis=-1, keepdims=True)

n_classes, n_samples, feature_dim = id_data.shape
Ks = [2, 3, 4, 7, 12, 20, 40, 80, 88]
seeds = list(range(10))
scoring_mode = "min_distance"  # or "softmax"
# scoring_mode = "softmax"
results_dict = {}

def compute_scores(X, prototypes, mode):
    dists = np.linalg.norm(X[:, None, :] - prototypes[None, :, :], axis=2)  # (N, K)
    if mode == "min_distance":
        return np.min(dists, axis=1)
    elif mode == "softmax":
        logits = -dists
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        max_softmax = np.max(softmax, axis=1)
        return -max_softmax
    else:
        raise ValueError(f"Unknown scoring mode: {mode}")

for K in Ks:
    print(f"Processing K = {K}")
    aurocs = []

    for seed in seeds:
        np.random.seed(seed)

        # Define OOD and ID classes
        ood_classes = list(range(10 * seed, 10 * (seed + 1)))
        available_classes = list(set(range(100)) - set(ood_classes))

        test_classes = np.random.choice(available_classes, size=2, replace=False)
        remaining_classes = list(set(available_classes) - set(test_classes))
        id_classes = list(test_classes) + list(np.random.choice(remaining_classes, size=K - 2, replace=False))

        # Gather ID samples
        X_id = []
        y_id = []
        for i, clss in enumerate(id_classes):
            samples = id_data[clss][:20]
            X_id.append(samples)
            y_id.extend([i] * len(samples))
        X_id = np.vstack(X_id)
        y_id = np.array(y_id)

        # Create test set (ID samples from test classes only)
        X_test_id = []
        for cls in test_classes:
            X_test_id.append(id_data[cls][20:50])
        X_test_id = np.vstack(X_test_id)

        X_ood = []
        for i, clss in enumerate(test_classes):
            samples = ood_data[clss]
            X_ood.append(samples)
        X_ood = np.vstack(X_ood)

        # Compute class prototypes
        prototypes = np.zeros((K, feature_dim))
        for i in range(K):
            class_samples = X_id[y_id == i]
            prototypes[i] = np.mean(class_samples, axis=0)

        # Compute scores
        scores_id = compute_scores(X_test_id, prototypes, mode=scoring_mode)
        scores_ood = compute_scores(X_ood, prototypes, mode=scoring_mode)

        labels = np.concatenate([np.zeros_like(scores_id), np.ones_like(scores_ood)])
        scores = np.concatenate([scores_id, scores_ood])
        auroc = roc_auc_score(labels, scores)
        aurocs.append(auroc)

    results_dict[K] = (np.mean(aurocs), np.std(aurocs))

# Plotting
mean_aurocs = [results_dict[K][0] for K in Ks]
std_aurocs = [results_dict[K][1] for K in Ks]

plt.figure(figsize=(8, 8))
bars = plt.bar(np.arange(len(Ks)),
               [val if not np.isnan(val) else 0 for val in mean_aurocs],
               yerr=[err if not np.isnan(err) else 0 for err in std_aurocs],
               capsize=8, color='steelblue', edgecolor='black')

for bar, val in zip(bars, mean_aurocs):
    if np.isnan(val):
        bar.set_visible(False)

plt.xlabel('Number of Classes (K)')
plt.ylabel('AUROC for OOD Detection')
plt.title(f'AUROC vs K using {scoring_mode} on CIFAR-100 (OOD = SVHN)')

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
