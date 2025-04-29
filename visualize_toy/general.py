import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
class1 = np.random.normal(loc=[-2, 0], scale=0.8, size=(100, 2))
class2 = np.random.normal(loc=[2, 0], scale=0.8, size=(100, 2))
ood_sample = np.array([[0, 4]])  # far OOD

X = np.vstack([class1, class2, ood_sample])
y = np.array([0]*100 + [1]*100 + [2])  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[:200])
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
clf.fit(X_scaled, y[:200])

ood_scaled = scaler.transform(ood_sample)
probs = clf.predict_proba(ood_scaled)
print("Softmax probability for OOD sample:", probs)

plt.figure(figsize=(7, 7))
plt.scatter(class1[:, 0], class1[:, 1], label='Class 1 (ID)', alpha=0.6, c='blue')
plt.scatter(class2[:, 0], class2[:, 1], label='Class 2 (ID)', alpha=0.6, c='green')
plt.scatter(ood_sample[0, 0], ood_sample[0, 1], label='OOD Sample', color='red', edgecolors='k', s=100, marker='X')

plt.title("General Setting: OOD Sample Has High Softmax Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("general_setting.png", dpi=300)
plt.show()
