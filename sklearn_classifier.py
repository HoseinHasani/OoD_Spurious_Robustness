import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
from dataset import GaussianDataset

N_train = 2000

dataset = GaussianDataset()



x_train = dataset.x[:N_train]
y_train = dataset.y[:N_train]
g_train = dataset.g[:N_train]


x_eval = dataset.x[N_train:]
y_eval = dataset.y[N_train:]
g_eval = dataset.g[N_train:]

ood_groups = dataset.o

N_class = len(set(y_train))
N_group = len(set(g_train))

train_group_inds = [np.argwhere(g_train == k).ravel() for k in range(N_group)]
eval_group_inds = [np.argwhere(g_eval == k).ravel() for k in range(N_group)]

clf = LogisticRegression(random_state=0)
clf.fit(x_train, y_train)

preds = clf.predict(x_eval)
eval_acc = 100 * accuracy_score(y_eval, preds)

g_eval_accs = []
for k in range(N_group):
    preds = clf.predict(x_eval[eval_group_inds[k]])
    g_acc = 100 * accuracy_score(y_eval[eval_group_inds[k]], preds)
    g_eval_accs.append(g_acc)
    
ood_probs = [clf.predict_proba(samples)[:, 0].mean() for samples in ood_groups]
    
print('************')
print('Total Accuracy:', np.round(eval_acc, 1))
print('Groups Accuracy:', np.round(g_eval_accs, 1))
print('OoD Probs:', ood_probs)
print('************')

lim=4

xx, yy = np.mgrid[-lim/2:lim/2:.01, -lim:lim:.01]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)


f, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu", vmin=0, vmax=1)
ax_c = f.colorbar(contour)
ax_c.set_label("$P(y = 1)$")
ax_c.set_ticks([0, .25, .5, .75, 1])

ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.scatter(ood_groups[0][:, 0], ood_groups[0][:, 1], c='k', s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.scatter(ood_groups[0][:, 0], ood_groups[1][:, 1], c='k', s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.set(aspect="equal",
       xlim=(-lim/2, lim/2), ylim=(-lim, lim),
       xlabel="$X_1$", ylabel="$X_2$")
    
clf = GaussianNB()
clf.fit(x_train, g_train)

preds = clf.predict(x_eval)
eval_acc = 100 * accuracy_score(g_eval, preds)
    
ood_probs = [clf.predict_proba(samples)[:, 0].mean() for samples in ood_groups]
    
print('************')
print('Total Accuracy:', np.round(eval_acc, 1))
print('OoD Probs:', ood_probs)
print('************')





