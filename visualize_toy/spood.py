import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams

# Style settings for paper figures
rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 160
})

np.random.seed(11)
majority2 = np.random.normal(loc=[2, -2], scale=0.55, size=(440, 2))
minority2 = np.random.normal(loc=[-2, -2], scale=0.65, size=(40, 2))
majority1 = np.random.normal(loc=[-2, 2], scale=0.55, size=(440, 2))
minority1 = np.random.normal(loc=[2, 2], scale=0.65, size=(40, 2))

sp_ood_sample = np.array([[4.5, -0.07]])
id_sample = np.array([[-2.63, -0.55]])

X = np.vstack([majority1, minority1, majority2, minority2])
y = np.array([0]*480 + [1]*480)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
clf.fit(X_scaled, y)

samples = np.vstack([sp_ood_sample, id_sample])
samples_scaled = scaler.transform(samples)
probs = clf.predict_proba(samples_scaled)

print("Softmax probability for spurious OOD:", probs[0])
print("Softmax probability for near-boundary ID sample:", probs[1])

plt.figure(figsize=(6, 6))

plt.scatter(majority1[:250, 0], majority1[:250, 1], label='Class 1 - Majority', alpha=0.65, c='tab:blue', s=25)
plt.scatter(minority1[:, 0], minority1[:, 1], label='Class 1 - Minority', alpha=0.40, c='tab:blue', s=25)
plt.scatter(majority2[:250, 0], majority2[:250, 1], label='Class 2 - Majority', alpha=0.65, c='tab:red', s=25)
plt.scatter(minority2[:, 0], minority2[:, 1], label='Class 2 - Minority', alpha=0.40, c='tab:red', s=25)

plt.scatter(sp_ood_sample[0, 0], sp_ood_sample[0, 1], label='Spurious OOD', color='purple',
            edgecolors='k', s=100, marker='s', linewidths=1.2)
plt.text(sp_ood_sample[0, 0] - 2.75, sp_ood_sample[0, 1] + 0.17,
         f"Max Softmax: {probs[0].max():.2f}", color='purple', fontsize=11, fontweight='bold')

plt.scatter(id_sample[0, 0], id_sample[0, 1], label='ID Sample', color='#ff7f0e',
            edgecolors='k', s=100, marker='o', linewidths=1.2)
plt.text(id_sample[0, 0] + 0.7, id_sample[0, 1],
         f"Max Softmax: {probs[1].max():.2f}", color='#ff7f0e', fontsize=11, fontweight='bold')

xx, yy = np.meshgrid(np.linspace(-4.55, 4.55, 400), np.linspace(-4.55, 4.55, 400))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)
Z = clf.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
contour = plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--')
plt.clabel(contour, inline=True, fmt='Decision Boundary', fontsize=9)

plt.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

plt.title("Logistic Regression")
plt.xlabel("Spurious Feature")
plt.ylabel("Core Feature")
plt.legend(loc='upper left')
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig("toy_spood_logistic.pdf")
plt.show()
