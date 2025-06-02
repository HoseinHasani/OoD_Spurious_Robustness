import numpy as np
import matplotlib.pyplot as plt
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
class1 = np.random.normal(loc=[-0.01, -2], scale=0.7, size=(360, 2))
class2 = np.random.normal(loc=[0.01, 2], scale=0.7, size=(360, 2))

ood_sample = np.array([[3.5, -0.05]])
id_sample = np.array([[-0.5, 0.077]])

proto1 = np.mean(class1, axis=0)
proto2 = np.mean(class2, axis=0)
prototypes = np.stack([proto1, proto2])
proto_colors = ['tab:red', 'tab:blue']

def distance(z, p):
    return np.sqrt(np.sum((z - p)**2))

samples = np.vstack([ood_sample, id_sample])
distances = np.array([[distance(s, p) for p in prototypes] for s in samples])
softmax_scores = np.exp(-distances) / np.sum(np.exp(-distances), axis=1, keepdims=True)

plt.figure(figsize=(6, 6))
plt.scatter(class1[:, 0], class1[:, 1], label='Class 1 (ID)', alpha=0.55, c='tab:red', s=25)
plt.scatter(class2[:, 0], class2[:, 1], label='Class 2 (ID)', alpha=0.55, c='tab:blue', s=25)

plt.scatter(proto1[0], proto1[1], marker='*', color='tab:red', s=180, edgecolors='k', linewidths=1.2, label='Prototype 1')
plt.scatter(proto2[0], proto2[1], marker='*', color='tab:blue', s=180, edgecolors='k', linewidths=1.2, label='Prototype 2')

plt.scatter(ood_sample[0, 0], ood_sample[0, 1], label='OOD Sample', color='purple',
            edgecolors='k', s=100, marker='s', linewidths=1.2)
plt.scatter(id_sample[0, 0], id_sample[0, 1], label='ID Sample', color='#ff7f0e',
            edgecolors='k', s=100, marker='o', linewidths=1.2)

for i, (sample, label, marker_color) in enumerate(zip(samples, ['OOD', 'ID'], ['purple', '#ff7f0e'])):
    nearest_idx = np.argmin(distances[i])
    nearest_proto = prototypes[nearest_idx]
    proto_color = proto_colors[nearest_idx]
    dist_val = distances[i][nearest_idx]
    softmax_val = softmax_scores[i][nearest_idx]

    plt.plot([sample[0], nearest_proto[0]], [sample[1], nearest_proto[1]],
             linestyle='--', color=proto_color, linewidth=1.3)

    shiftx = -2.96 if label == 'ID' else -1.2
    shifty = 0.1 if label == 'ID' else 0.24
    plt.text(sample[0] + shiftx, sample[1] + shifty,
             f"Min Distance: {dist_val:.2f}",
             color=marker_color, fontsize=11, fontweight='bold')

plt.title("Prototypical Classification")
plt.legend(loc='upper right')
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.ylabel("Core Feature")
plt.tight_layout()
plt.savefig("toy_ood_prototypical.pdf")
plt.show()
