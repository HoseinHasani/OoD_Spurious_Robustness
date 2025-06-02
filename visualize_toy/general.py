import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams

rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 160
})

np.random.seed(4)

# Generate synthetic in-distribution data for two classes
class1 = np.random.normal(loc=[-0.01, -2], scale=0.7, size=(360, 2))  # Below cluster
class2 = np.random.normal(loc=[0.01, 2], scale=0.7, size=(360, 2))   # Above cluster

# Define an out-of-distribution (OOD) sample far from ID clusters
ood_sample = np.array([[3.5, -0.05]])
# Define an in-distribution (ID) sample near the decision boundary
id_sample = np.array([[-0.5, 0.09]])

# Combine data and labels
X = np.vstack([class1, class2])
y = np.array([0]*len(class1) + [1]*len(class2))

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression classifier
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
clf.fit(X_scaled, y)

# Scale test samples and predict softmax probabilities
samples = np.vstack([ood_sample, id_sample])
samples_scaled = scaler.transform(samples)
probs = clf.predict_proba(samples_scaled)

print("Softmax probability for OOD sample:", probs[0])
print("Softmax probability for near-boundary ID sample:", probs[1])

plt.figure(figsize=(6, 6))

# Plot ID class points with requested colors
plt.scatter(class1[:, 0], class1[:, 1], label='Class 1 (ID)', alpha=0.55, c='tab:red', s=25)
plt.scatter(class2[:, 0], class2[:, 1], label='Class 2 (ID)', alpha=0.55, c='tab:blue', s=25)

# Plot OOD and near-boundary ID samples with requested markers and colors
plt.scatter(ood_sample[0, 0], ood_sample[0, 1], label='OOD Sample', color='purple',
            edgecolors='k', s=100, marker='s', linewidths=1.2)  # square marker 's'
plt.scatter(id_sample[0, 0], id_sample[0, 1], label='ID Sample', color='#ff7f0e',
            edgecolors='k', s=100, marker='o', linewidths=1.2)

# Annotate max softmax probabilities
plt.text(ood_sample[0, 0] - 1.9, ood_sample[0, 1] + 0.2,
         f"Max Softmax: {probs[0].max():.2f}", color='purple', fontsize=11, fontweight='bold')
plt.text(id_sample[0, 0] - 2.77, id_sample[0, 1] + 0.2,
         f"Max Softmax: {probs[1].max():.2f}", color='#ff7f0e', fontsize=11, fontweight='bold')

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(-3.4, 4, 400), np.linspace(-4, 4, 400))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)
Z = clf.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
contour = plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--')
plt.clabel(contour, inline=True, fmt='Decision Boundary', fontsize=9)

# Plot grid lines and axes lines
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

plt.title("Logistic Regression")
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', linewidth=0.5)
plt.axis('equal')
plt.ylabel("Core Feature")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig("toy_ood_logistic.pdf")
plt.show()
