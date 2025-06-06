import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

np.random.seed(42)

K = 30
N_per_class = 200
center_std = 2.0 + K / 30
radius = 8
n_ood = 500



def generate_distant_centers(K, center_std=2.0, min_dist=0.4, max_attempts=1000):
    centers = []
    attempts = 0
    while len(centers) < K and attempts < max_attempts:
        candidate = np.random.randn(2) * center_std
        if all(np.linalg.norm(candidate - c) >= min_dist for c in centers):
            if np.linalg.norm(candidate) < 0.8 * radius:
                centers.append(candidate)
        attempts += 1
    if len(centers) < K:
        print(f"Could not find {K} well-separated centers after {max_attempts} attempts.")
    return np.array(centers)

centers = generate_distant_centers(K=K, center_std=center_std, min_dist=0.4)


decay = 1 / K
class_stds = np.random.uniform(0.1 + 0.6 * decay, 0.4 + decay, size=K)

X_id = []
y_id = []

for i, (c, std) in enumerate(zip(centers, class_stds)):
    samples = np.random.randn(N_per_class, 2) * std + c
    X_id.append(samples)
    y_id.extend([i] * N_per_class)

X_id = np.vstack(X_id)
y_id = np.array(y_id)

angles = 3 * np.pi * np.random.rand(n_ood)
radii = radius + 0.5 * np.random.randn(n_ood)
X_ood = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

# clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
clf = LogisticRegression(solver='lbfgs')
clf.fit(X_id, y_id)

x_min, x_max = -10, 10
y_min, y_max = -10, 10
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = clf.predict_proba(grid)
preds = np.argmax(probs, axis=1)


base_cmap = plt.get_cmap('hsv')
class_colors = [base_cmap(i / K) for i in range(K)]
region_cmap = ListedColormap(class_colors)


plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, preds.reshape(xx.shape), 
             levels=np.arange(K + 1) - 0.5, 
             cmap=region_cmap, alpha=0.2)

for i in range(K):
    plt.scatter(X_id[y_id == i][:, 0], X_id[y_id == i][:, 1],
                color=class_colors[i], label=f'Class {i}', s=15, alpha=0.6)

plt.scatter(X_ood[:, 0], X_ood[:, 1], c='black', marker='x', label='OOD samples', alpha=0.6)

plt.legend()
plt.title('In-Distribution Classes (Random Centers/Std) and OOD Ring with Softmax Boundaries')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()