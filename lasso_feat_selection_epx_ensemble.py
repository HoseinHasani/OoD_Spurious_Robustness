import torch
from torch import nn
import torch.nn.functional as F
import pickle
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
n_feat_f = 10
n_iteration = 10
n_feats = 1024
n_steps = 500
n_data_eval = 1000
torch.manual_seed(seed)
np.random.seed(seed)
g_batch_size = 64
gamma = 0.01

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#!cp ./drive/MyDrive/grouped_embs.pkl .
with open('grouped_embs.pkl', 'rb') as f:
    grouped_embs = pickle.load(f)
    
    
grouped_embs['bald'] = np.concatenate([grouped_embs['woman_bald'], grouped_embs['man_bald']])
del grouped_embs['man_bald']
del grouped_embs['woman_bald']
grouped_prototypes = {group: embs.mean(axis=0, keepdims=True) for group, embs in grouped_embs.items()}
all_embs = np.concatenate(list(grouped_embs.values()))
all_prototypes = np.concatenate(list(grouped_prototypes.values()))
y_true = np.concatenate([[i] * len(group_embs) for i, group_embs in enumerate(grouped_embs.values())])
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

def train(names_0, names_1, feat_inds, plot=True):
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

for k in range(n_iteration):
    mlp_sp = train(['woman_black', 'woman_blond'], ['man_black', 'man_blond'], feat_inds=np.arange(n_feats))
            
    weight = next(mlp_sp.parameters()).detach().cpu().numpy()
    woman_feats = np.argsort(weight[0])[-n_feat_f:]
    man_feats = np.argsort(weight[1])[-n_feat_f:]
    negative_feats = {'woman': woman_feats, 'man': man_feats}
    merged_sp_feats = list(set(np.concatenate([woman_feats, man_feats])))

####################################################
    
non_sp_feats = np.delete(np.arange(n_feats), merged_sp_feats)
    
print('Train core feature classifier:')
for k in range(n_iteration):
    mlp_core = train(['man_blond', 'woman_blond'], ['man_black', 'woman_black'], feat_inds=non_sp_feats)
             
    weight = next(mlp_core.parameters()).detach().cpu().numpy()
    blond_feats = np.argsort(weight[0])[-n_feat_f:]
    black_feats = np.argsort(weight[1])[-n_feat_f:]
    core_feats = {'blond': blond_feats, 'black': black_feats}


####################################################

def calc_cos_dist2(embs, prototypes, prototype_name):
    if 'blond' in prototype_name:
        feat_inds = core_feats['blond']
    elif 'black' in prototype_name:
        feat_inds = core_feats['black']
    else:
        print('WRONG NAME!')
    embs = embs[:, non_sp_feats][:, feat_inds]
    prototypes = prototypes[:, non_sp_feats][:, feat_inds]

    embs_normalized = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    prototypes_normalized = prototypes / np.linalg.norm(prototypes, axis=-1, keepdims=True)
    cos_dist = (1 - (embs_normalized[:, None] * prototypes_normalized).sum(axis=-1)) / 2
    return cos_dist.squeeze()


grouped_cos_dist = {group: calc_cos_dist2(grouped_embs[group], grouped_prototypes[group], group) for group in list(grouped_embs.keys())[:-1]}


fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for group, ax in zip([group for group in grouped_embs if not 'bald' in group], axes):
    sns.kdeplot(grouped_cos_dist[group], label=group, palette=['red'], ax=ax)
    sns.kdeplot(calc_cos_dist2(grouped_embs['bald'], grouped_prototypes[group], group), label='ood', ax=ax)
    ax.legend()
    ax.set_title(group)
    
    
####################################################
    

merged_core_feats = list(set(np.concatenate([blond_feats, black_feats])))

data_np, lbl_np = prepare_data(['man_blond', 'woman_blond'], ['man_black', 'woman_black'], 1000, np.arange(1000))
x_train = data_np[:, non_sp_feats][:, merged_core_feats]
y_train = lbl_np

clf = LogisticRegression()
clf.fit(x_train, y_train)

data_np, lbl_np = prepare_data(['man_blond', 'woman_blond'], ['man_black', 'woman_black'], 500, np.arange(-500,0))
x_eval = data_np[:, non_sp_feats][:, merged_core_feats]
y_eval = lbl_np

preds = clf.predict(x_eval)
eval_acc = 100 * (preds == y_eval).mean()

print('BLOND / BLACK ACC:', eval_acc)

data_np, lbl_np = prepare_data(['woman_black', 'woman_blond'], ['man_black', 'man_blond'], 1000, np.arange(1000))
x_train = data_np[:, non_sp_feats][:, merged_core_feats]
y_train = lbl_np

clf = LogisticRegression()
clf.fit(x_train, y_train)

data_np, lbl_np = prepare_data(['woman_black', 'woman_blond'], ['man_black', 'man_blond'], 500, np.arange(-500,0))
x_eval = data_np[:, non_sp_feats][:, merged_core_feats]
y_eval = lbl_np

preds = clf.predict(x_eval)
eval_acc = 100 * (preds == y_eval).mean()

print('MAN / WOMAN ACC:', eval_acc)

####################################################