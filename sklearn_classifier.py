import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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


