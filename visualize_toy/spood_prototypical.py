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

np.random.seed(11)
majority1 = np.random.normal(loc=[-2, -2], scale=0.55, size=(360, 2))
minority1 = np.random.normal(loc=[2, -2], scale=0.65, size=(40, 2))
majority2 = np.random.normal(loc=[2, 2], scale=0.55, size=(360, 2))
minority2 = np.random.normal(loc=[-2, 2], scale=0.65, size=(40, 2))

sp_ood_sample = np.array([[4.5, 0.07]])
id_sample = np.array([[-2.63, 0.47]])
samples = np.vstack([sp_ood_sample, id_sample])

proto_m1 = np.mean(majority1, axis=0)
proto_m2 = np.mean(minority1, axis=0)
proto_m3 = np.mean(majority2, axis=0)
proto_m4 = np.mean(minority2, axis=0)

prototypes = np.stack([proto_m1, proto_m2, proto_m3, proto_m4])
proto_colors = ['tab:blue', 'tab:blue', 'tab:red', 'tab:red']

def distance(z, p):
    return np.sqrt(np.sum((z - p)**2))

all_distances = np.array([[distance(s, p) for p in prototypes] for s in samples])

grouped_distances = np.array([
    [np.min(dist[:2]), np.min(dist[2:])] for dist in all_distances
])

# Softmax over 2 class-prototype distances
softmax_scores = np.exp(-grouped_distances) / np.sum(np.exp(-grouped_distances), axis=1, keepdims=True)

plt.figure(figsize=(6, 6))
plt.scatter(majority1[:, 0], majority1[:, 1], label='Class 1 - Majority', alpha=0.65, c='tab:blue', s=25)
plt.scatter(minority1[:, 0], minority1[:, 1], label='Class 1 - Minority', alpha=0.35, c='tab:blue', s=25)
plt.scatter(majority2[:, 0], majority2[:, 1], label='Class 2 - Majority', alpha=0.65, c='tab:red', s=25)
plt.scatter(minority2[:, 0], minority2[:, 1], label='Class 2 - Minority', alpha=0.35, c='tab:red', s=25)

for i, proto in enumerate(prototypes):
    plt.scatter(proto[0], proto[1], marker='*', color=proto_colors[i], s=180,
                edgecolors='k', linewidths=1.2, label=f'Prototype {i+1}')

plt.scatter(sp_ood_sample[0, 0], sp_ood_sample[0, 1], label='Spurious OOD', color='purple',
            edgecolors='k', s=100, marker='s', linewidths=1.2)  # purple square
plt.scatter(id_sample[0, 0], id_sample[0, 1], label='ID Sample', color='#ff7f0e',
            edgecolors='k', s=100, marker='o', linewidths=1.2)  # same orange circle

for i, (sample, label, color) in enumerate(zip(samples, ['Spurious OOD', 'ID'], ['purple', '#ff7f0e'])):
    class1_idx = np.argmin(all_distances[i][:2])
    class2_idx = np.argmin(all_distances[i][2:])
    min_class1_proto = prototypes[class1_idx]
    min_class2_proto = prototypes[2 + class2_idx]
    
    if grouped_distances[i][0] < grouped_distances[i][1]:
        nearest_proto = min_class1_proto
        nearest_color = 'tab:blue'
        softmax_val = softmax_scores[i][0]
        dist_val = grouped_distances[i][0]
    else:
        nearest_proto = min_class2_proto
        nearest_color = 'tab:red'
        softmax_val = softmax_scores[i][1]
        dist_val = grouped_distances[i][1]

    plt.plot([sample[0], nearest_proto[0]], [sample[1], nearest_proto[1]],
             linestyle='--', color=nearest_color, linewidth=1.3)
    
    shiftx = .2 if label == 'ID' else -2.9
    shifty = -0.5 if label == 'ID' else -0.28
    plt.text(sample[0] + shiftx, sample[1] + shifty,
             f"Min Distance: {dist_val:.2f}\nMax Softmax: {softmax_val:.2f}",
             color=color, fontsize=11, fontweight='bold')

plt.title("Prototypical Classification")
plt.xlabel("Spurious Feature")
plt.ylabel("Core Feature")
plt.axis('equal')
plt.xticks([])
plt.yticks([])
# plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("toy_spood_prototypical.pdf")
plt.show()
