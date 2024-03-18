import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset import GaussianDataset3D

#%matplotlib qt

dataset = GaussianDataset3D(3)
g_embs = dataset.grouped_embs



def draw_point_cloud3D(ax, data, label, color, alpha=0.15):
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color=color, label=label, alpha=alpha, s=2)

def plot_prototype3D(ax, data, label, color):
    ax.scatter(data[0], data[1], data[2], color=color, label=label, alpha=1, s=12)
    
def draw_arrow3D(ax, arrow, label, color, linestyle):
    ax.quiver(0, 0, 0, arrow[0], arrow[1], arrow[2],
              label=label, color=color, linestyle=linestyle, linewidth=2)
    
   
figsize = 10
fig = plt.figure(figsize=(figsize, figsize))

ax = fig.add_subplot(projection='3d')

draw_point_cloud3D(ax, g_embs['0_0'], 'maj 0', 'tab:blue')
draw_point_cloud3D(ax, g_embs['0_1'], 'min 0', 'tab:green')
draw_point_cloud3D(ax, g_embs['1_0'], 'maj 1', 'tab:orange')
draw_point_cloud3D(ax, g_embs['1_1'], 'min 1', 'tab:red')

plot_prototype3D(ax, g_embs['0_0'].mean(0), 'maj 0 - prototype', 'blue')
plot_prototype3D(ax, g_embs['0_1'].mean(0), 'maj 0 - prototype', 'green')
plot_prototype3D(ax, g_embs['1_0'].mean(0), 'maj 1 - prototype', 'orange')
plot_prototype3D(ax, g_embs['1_1'].mean(0), 'maj 1 - prototype', 'red')

draw_arrow3D(ax, dataset.core_ax, 'core axis', 'red', 'solid')
draw_arrow3D(ax, dataset.sp_ax, 'spurious axis', 'orange', 'dashed')

draw_point_cloud3D(ax, dataset.o[0], 'OoD', 'tab:gray')

fig.tight_layout()
plt.legend()
