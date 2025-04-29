import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
majority1 = np.random.normal(loc=[-2, 0], scale=0.6, size=(80, 2))
minority1 = np.random.normal(loc=[-2, 2], scale=0.6, size=(20, 2))
majority2 = np.random.normal(loc=[2, 0], scale=0.6, size=(80, 2))
minority2 = np.random.normal(loc=[2, -2], scale=0.6, size=(20, 2))

sp_ood = np.array([[-2.2, -1.0]])

X = np.vstack([majority1, minority1, majority2, minority2])
y = np.array([0]*100 + [1]*100)  # Class labels

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
clf.fit(X_scaled, y)

sp_ood_scaled = scaler.transform(sp_ood)
probs = clf.predict_proba(sp_ood_scaled)
print("Softmax probability for spurious OOD:", probs)

plt.figure(figsize=(7, 7))
plt.scatter(majority1[:, 0], majority1[:, 1], label='Class 1 - Majority', alpha=0.6, c='dodgerblue')
plt.scatter(minority1[:, 0], minority1[:, 1], label='Class 1 - Minority', alpha=0.6, c='lightskyblue')
plt.scatter(majority2[:, 0], majority2[:, 1], label='Class 2 - Majority', alpha=0.6, c='limegreen')
plt.scatter(minority2[:, 0], minority2[:, 1], label='Class 2 - Minority', alpha=0.6, c='palegreen')
plt.scatter(sp_ood[0, 0], sp_ood[0, 1], label='Spurious OOD', color='crimson', edgecolors='k', s=120, marker='X')

plt.title("Spurious OOD: Near Majority but Semantically Different")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("spurious_setting.png", dpi=300)
plt.show()
