import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from dataset import GaussianDataset
import utils
import os

N_train = 2000

pic_path = 'results/sklearn_pics/'
os.makedirs(pic_path, exist_ok=True)

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
    
ood_probs = [np.round(clf.predict_proba(samples).max(-1).mean(), 3) for samples in ood_groups]
    
print('************')
print('Discriminative Approach:')
print('Total Accuracy:', np.round(eval_acc, 1))
print('Groups Accuracy:', np.round(g_eval_accs, 1))
print('OoD Probs (first group, second group):', tuple(ood_probs))
print('************')

utils.visualize_clf_boundary(clf, x_train, y_train, ood_groups, 'Logistic Regression', pic_path)

clf = GaussianNB()
clf.fit(x_train, g_train)

preds = clf.predict(x_eval)
eval_acc = 100 * accuracy_score(g_eval, preds)
    
g_eval_accs = []
for k in range(N_group):
    preds = clf.predict(x_eval[eval_group_inds[k]])
    g_acc = 100 * accuracy_score(g_eval[eval_group_inds[k]], preds)
    g_eval_accs.append(g_acc)
    
    
class_conditional_probs = [utils.calc_class_conditional_probs(samples, clf) for samples in ood_groups]
ood_probs = [np.round(class_conditional_probs[k].max(-1).mean(), 3) for k in range(len(ood_groups))]
    


print('************')
print('Generative Approach:')
print('Total Accuracy:', np.round(eval_acc, 1))
print('Groups Accuracy:', np.round(g_eval_accs, 1))
print('OoD Probs (first group, second group):', tuple(ood_probs))
print('************')

utils.visualize_clf_boundary(clf, x_train, g_train//2, ood_groups, 'GaussianNB Classifier', pic_path)
utils.visualize_OoD_dist(clf, x_train, ood_groups, 'In-Distribution PDF', pic_path)




