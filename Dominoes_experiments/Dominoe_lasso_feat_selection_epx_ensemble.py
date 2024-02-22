import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import os
import tqdm
import warnings
warnings.filterwarnings("ignore")


# set seed
seed = 8
n_feat_per_iter_sp = 200
n_feat_per_iter_core = 1

n_iteration = 7
n_feats = 1024
n_steps = 500
n_data_eval = 1000
torch.manual_seed(seed)
np.random.seed(seed)
g_batch_size = 64
gamma = 0.01

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


grouped_embs = np.load('Dominoes_grouped_embs.npy', allow_pickle=True).item()
    
    
grouped_prototypes = {group: embs.mean(axis=0, keepdims=True) for group, embs in grouped_embs.items()}
all_embs = np.concatenate(list(grouped_embs.values()))
all_prototypes = np.concatenate(list(grouped_prototypes.values()))
group_names = list(grouped_embs.keys())



class MLP(nn.Module):
    def __init__(self, n_feat=n_feats, n_out=2):
        super().__init__()
        self.layer = nn.Linear(n_feat, n_out)

    def forward(self, x):
        return self.layer(x)
    
    
####################################################

def prepare_class_data(group_names, len_g, lbl_val, inds=None):
    data_list = []
    for name in group_names:
        if inds is None:
            g_inds = np.random.choice(len(grouped_embs[name]), len_g, replace=False)
        else:
            g_inds = inds
            
        assert len(g_inds) == len_g
        
        data_list.append(grouped_embs[name][g_inds])
    
    data = np.concatenate(data_list)
    labels = lbl_val * np.ones(len(group_names) * len_g)
    
    return data, labels

def prepare_data(names_0, names_1, len_g, inds=None):
    data0, lbl0 = prepare_class_data(names_0, len_g, 0, inds)
    data1, lbl1 = prepare_class_data(names_1, len_g, 1, inds)
    
    data_np = np.concatenate([data0, data1])
    lbl_np = np.concatenate([lbl0, lbl1]).astype(int)
    
    return data_np, lbl_np

def train(names_0, names_1, feat_inds, plot=False):
    mlp = MLP(n_feat=len(feat_inds)).to(device)  
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        
    mlp.train()
        
    for e in tqdm.tqdm(range(n_steps)):
        
        data_np, lbl_np = prepare_data(names_0, names_1, g_batch_size)
        
        data = torch.tensor(data_np, dtype=torch.float32).to(device)[:, feat_inds]  
        lbl = torch.tensor(lbl_np, dtype=torch.long).to(device)  
        
        optimizer.zero_grad()
        logits = mlp(data)
        weight = next(mlp.parameters())
        l1_loss = torch.abs(weight).sum()
        loss = loss_function(logits, lbl) + gamma * l1_loss
        loss.backward()
        optimizer.step()
        
        if e % int(n_steps / 5) == 0:
            with torch.no_grad():
                if plot:
                    plt.figure()
                    weight = next(mlp.parameters()).detach().cpu().numpy()
                    plt.hist(np.abs(weight.ravel()), 100)
                    plt.title(str(e))
                
                data_np, lbl_np = prepare_data(names_0, names_1, n_data_eval)
                
                data_eval = torch.tensor(data_np, dtype=torch.float32).to(device)[:, feat_inds]    
                lbl_eval = torch.tensor(lbl_np, dtype=torch.long).to(device)  
                
                pred = mlp(data_eval).max(-1)[1]
                acc = (pred == lbl_eval).float().mean().item()
                print('train acc:', acc)
            
    return mlp

####################################################

print('Train spurious feature classifier:')

sp_feats = []

for k in range(n_iteration):
    if k:
        cur_feats = np.delete(np.arange(n_feats), np.concatenate(sp_feats))
    else:
        cur_feats = np.arange(n_feats)
        
    mlp_sp = train(['0_airplane', '0_car'], ['1_airplane', '1_car'], feat_inds=cur_feats)
            
    weight = next(mlp_sp.parameters()).detach().cpu().numpy()
    zero_feats = np.argsort(weight[0])[-n_feat_per_iter_sp:]
    one_feats = np.argsort(weight[1])[-n_feat_per_iter_sp:]
    negative_feats = {'zero': zero_feats, 'one': one_feats}
    merged_sp_feats = list(set(np.concatenate([zero_feats, one_feats])))
    sp_feats.append(merged_sp_feats)
    
sp_feats = np.unique(np.concatenate(sp_feats))

####################################################
    
non_sp_feats = np.delete(np.arange(n_feats), sp_feats)
    
core_feats = []

print('Train core feature classifier:')
for k in range(n_iteration):
    if k:
        cur_feats = np.delete(non_sp_feats, np.concatenate(core_feats))
    else:
        cur_feats = non_sp_feats
    mlp_core = train(['0_airplane', '1_airplane'], ['0_car', '1_car'], feat_inds=cur_feats)
             
    weight = next(mlp_core.parameters()).detach().cpu().numpy()
    airplane_feats = np.argsort(weight[0])[-n_feat_per_iter_core:]
    car_feats = np.argsort(weight[1])[-n_feat_per_iter_core:]
    positive_feats = {'airplane': airplane_feats, 'car': car_feats}
    merged_core_feats = list(set(np.concatenate([airplane_feats, car_feats])))
    core_feats.append(merged_core_feats)

core_feats = np.unique(np.concatenate(core_feats))

print('Feature sizes:', len(sp_feats), len(core_feats))

####################################################


    


data_np, lbl_np = prepare_data(['0_airplane', '0_car'], ['1_airplane', '1_car'], 1000, np.arange(1000))
x_train = data_np[:, non_sp_feats][:, core_feats]
y_train = lbl_np

clf = LogisticRegression()
clf.fit(x_train, y_train)

data_np, lbl_np = prepare_data(['0_airplane', '0_car'], ['1_airplane', '1_car'], 500, np.arange(-500,0))
x_eval = data_np[:, non_sp_feats][:, core_feats]
y_eval = lbl_np

preds = clf.predict(x_eval)
eval_acc = 100 * (preds == y_eval).mean()

print('ZERO / ONE ACC:', eval_acc)

data_np, lbl_np = prepare_data(['0_airplane', '1_airplane'], ['0_car', '1_car'], 1000, np.arange(1000))
x_train = data_np[:, non_sp_feats][:, core_feats]
y_train = lbl_np

clf = LogisticRegression()
clf.fit(x_train, y_train)

data_np, lbl_np = prepare_data(['0_airplane', '1_airplane'], ['0_car', '1_car'], 500, np.arange(-500,0))
x_eval = data_np[:, non_sp_feats][:, core_feats]
y_eval = lbl_np

preds = clf.predict(x_eval)
eval_acc = 100 * (preds == y_eval).mean()

print('AIRPLANE / CAR ACC:', eval_acc)

####################################################


def calc_cos_dist(embs, prototypes, prototype_name):
    embs = embs[:, non_sp_feats][:, core_feats]
    prototypes = prototypes[:, non_sp_feats][:, core_feats]

    embs_normalized = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    prototypes_normalized = prototypes / np.linalg.norm(prototypes, axis=-1, keepdims=True)
    cos_dist = (1 - (embs_normalized[:, None] * prototypes_normalized).sum(axis=-1)) / 2
    return cos_dist.squeeze()

ood_class_names = ['ship', 'truck']
selected_groups_names = ['0_airplane', '0_car', '1_airplane', '1_car']
selected_grouped_embs = {name: grouped_embs[name] for name in selected_groups_names}
grouped_cos_dist = {group: calc_cos_dist(grouped_embs[group], grouped_prototypes[group], group) for group in selected_groups_names}


fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for group, ax in zip([group for group in selected_grouped_embs], axes):
    sns.kdeplot(grouped_cos_dist[group], label=group, palette=['red'], ax=ax)
    sp_name = group[:2]
    ood_embs = np.concatenate([grouped_embs[sp_name + class_name] for class_name in ood_class_names])
    sns.kdeplot(calc_cos_dist(ood_embs, grouped_prototypes[group], group), label='ood', ax=ax)
    ax.legend()
    ax.set_title(group)
    
    
####################################################


