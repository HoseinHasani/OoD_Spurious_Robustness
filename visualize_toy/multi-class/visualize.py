import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

np.random.seed(42)

K = 4                    
N_per_class = 200        
std = 0.5                
radius = 6              
n_ood = 500              

theta = np.linspace(0, 2*np.pi, K, endpoint=False)
centers = np.stack([np.cos(theta), np.sin(theta)], axis=1)

X_id = []
y_id = []

for i, c in enumerate(centers):
    samples = np.random.randn(N_per_class, 2) * std + c
    X_id.append(samples)
    y_id.extend([i] * N_per_class)

X_id = np.vstack(X_id)
y_id = np.array(y_id)

angles = 2 * np.pi * np.random.rand(n_ood)
radii = radius + 0.5 * np.random.randn(n_ood)
X_ood = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
clf.fit(X_id, y_id)

x_min, x_max = -8, 8
y_min, y_max = -8, 8
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = clf.predict_proba(grid)
preds = np.argmax(probs, axis=1)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, preds.reshape(xx.shape), alpha=0.2, levels=np.arange(K+1)-0.5, cmap='tab10')

for i in range(K):
    plt.scatter(X_id[y_id == i][:, 0], X_id[y_id == i][:, 1], label=f'Class {i}', s=15, alpha=0.4)

plt.scatter(X_ood[:, 0], X_ood[:, 1], c='black', marker='x', label='OOD samples', alpha=0.6)

plt.legend()
plt.title('In-Distribution Classes and OOD Ring Samples with Softmax Decision Boundaries')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.axis('equal')
plt.show()
