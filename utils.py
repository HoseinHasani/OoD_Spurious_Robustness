import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
import numpy as np

sns.set(style="white")

def visualize_clf_boundary(clf, x_data, y_data, ood_groups, lim=4):

    xx, yy = np.mgrid[-lim/2:lim/2:.01, -lim:lim:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(grid)
    if probs.shape[-1] > 2:
        probs[:, -1] = probs[:, -2:].sum(-1)

    probs = probs[:, -1].reshape(xx.shape)
    
    
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu", vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])
    
    ax.scatter(x_data[:, 0], x_data[:, 1], c=y_data, s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)
    
    ax.scatter(ood_groups[0][:, 0], ood_groups[0][:, 1], c='k', s=50,
               edgecolor="white", linewidth=1)
    
    ax.scatter(ood_groups[1][:, 0], ood_groups[1][:, 1], c='k', s=50,
               edgecolor="white", linewidth=1)
    
    ax.set(aspect="equal",
           xlim=(-lim/2, lim/2), ylim=(-lim, lim),
           xlabel="$X_1$", ylabel="$X_2$")
    
    
def visualize_OoD_dist(clf, x_data, ood_groups, lim=4):

    xx, yy = np.mgrid[-lim/2:lim/2:.01, -lim:lim:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs_ = calc_class_conditional_probs(grid, clf)
    probs_ = probs_ / probs_.max(0)
    probs = np.array([probs_[:, k].reshape(xx.shape) for k in range(probs_.shape[-1])])
    probs = probs.sum(0)
    
    
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu", vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])
    
    ax.scatter(x_data[:, 0], x_data[:, 1], c='b', s=10,
               edgecolor="white", linewidth=.3, alpha=0.1)
    
    ax.scatter(ood_groups[0][:, 0], ood_groups[0][:, 1], c='k', s=20,
               edgecolor="white", linewidth=.5, alpha=0.3)
    
    ax.scatter(ood_groups[1][:, 0], ood_groups[1][:, 1], c='k', s=20,
               edgecolor="white", linewidth=.5, alpha=0.3)
    
    ax.set(aspect="equal",
           xlim=(-lim/2, lim/2), ylim=(-lim, lim),
           xlabel="$X_1$", ylabel="$X_2$")
    
    
def calc_class_conditional_probs(x, clf):
    means = clf.theta_
    stds = np.sqrt(clf.var_)
    
    class_MVNs = []
    for k in range(len(means)):
        MVN = multivariate_normal(means[k], np.diag(stds[k]))
        class_MVNs.append(MVN)
    
    probs = [MVN.pdf(x) for MVN in class_MVNs]
    
    return np.array(probs).T


