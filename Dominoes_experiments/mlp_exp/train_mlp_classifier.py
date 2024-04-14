import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
import warnings
warnings.filterwarnings("ignore")

normalize_embs = True

n_steps = 100
n_feats = 1024
batch_size = 128
sp_rate = 0.95


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_path = 'data'
train_dict0 = np.load(f'{data_path}/Dominoes_train_embs.npy', allow_pickle=True).item()
test_dict0 = np.load(f'{data_path}/Dominoes_test_embs.npy', allow_pickle=True).item()
all_dict = np.load('../Dominoes_grouped_embs.npy', allow_pickle=True).item()
sdfsdf

def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

if normalize_embs:
    train_dict = {key: normalize(train_dict0[key]) for key in train_dict0.keys()}
    test_dict = {key: normalize(test_dict0[key]) for key in test_dict0.keys()}
else:
    train_dict = train_dict0
    test_dict = test_dict0

names = ['automobile', 'truck']

l_maj = len(train_dict[f'0_{names[0]}'])
l_min = len(train_dict[f'1_{names[0]}'])

def prepare_train_data():
    ind_0_maj = np.random.choice(l_maj, size=int(sp_rate * batch_size), replace=False).ravel()
    ind_0_min = np.random.choice(l_min, size=int((1 - sp_rate) * batch_size), replace=False).ravel()

    ind_1_maj = np.random.choice(l_maj, size=int(sp_rate * batch_size), replace=False).ravel()
    ind_1_min = np.random.choice(l_min, size=int((1 - sp_rate) * batch_size), replace=False).ravel()
        
    data_np = np.concatenate([
                            train_dict[f'0_{names[0]}'][ind_0_maj],
                            train_dict[f'1_{names[0]}'][ind_0_min],
                            train_dict[f'1_{names[1]}'][ind_1_maj],
                            train_dict[f'0_{names[1]}'][ind_1_min],
                            ])
    
    lbl_np = np.concatenate([
                            np.zeros(len(ind_0_maj)),
                            np.ones(len(ind_0_min)),
                            np.ones(len(ind_1_maj)),
                            np.zeros(len(ind_1_min)),
                            ])
    
    return data_np, lbl_np
    

def visualize_correlations(embeddings, core_ax, sp_ax, value_dict=None, print_logs=True):
    
    c_vals = []
    c_vals_ood = []

    s_vals = []
    s_vals_ood = []
    
    for key in embeddings.keys():
        c_vals_ = np.abs(np.dot(embeddings[key], core_ax))
        s_vals_ = np.abs(np.dot(embeddings[key], sp_ax))
        
        if 'OOD' in key:
            if int(key[-1]) < 5:
                c_vals_ood.append(c_vals_)
                s_vals_ood.append(s_vals_)
        else:
            c_vals.append(c_vals_)
            s_vals.append(s_vals_)

    c_vals = np.concatenate(c_vals)
    c_vals_ood = np.concatenate(c_vals_ood)
    s_vals = np.concatenate(s_vals)
    s_vals_ood = np.concatenate(s_vals_ood)

    
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.hist(c_vals, 25, histtype='step', density=True, linewidth=2.5, label='embs', color='tab:blue')
    plt.hist(c_vals_ood, 25, histtype='step', density=True, linewidth=2.5, label='ood', color='tab:orange')
    plt.title('core alignment')
    plt.subplot(122)
    plt.hist(s_vals, 25, histtype='step', density=True, linewidth=2.5, label='embs', color='tab:blue')
    plt.hist(s_vals_ood, 25, histtype='step', density=True, linewidth=2.5, label='ood', color='tab:orange')
    plt.title('sp alignment')
    plt.legend()
    
    if value_dict is not None:
        plt.subplot(121)
        plt.hist(value_dict['c_vals'], 25, histtype='step', linestyle='dotted',
                 density=True, linewidth=2.5, label='embs (before)', color='tab:blue')
        plt.hist(value_dict['c_vals_ood'], 25, histtype='step', linestyle='dotted',
                 density=True, linewidth=2.5, label='ood (before)', color='tab:orange')
        plt.title('core alignment')
        plt.subplot(122)
        plt.hist(value_dict['s_vals'], 25, histtype='step', linestyle='dotted',
                 density=True, linewidth=2.5, label='embs (before)', color='tab:blue')
        plt.hist(value_dict['s_vals_ood'], 25, histtype='step', linestyle='dotted',
                 density=True, linewidth=2.5, label='ood (before)', color='tab:orange')
        plt.title('sp alignment')
        plt.legend()
    
    if print_logs:
        print(f'core coefs ratio: {np.mean(c_vals) / np.mean(c_vals_ood)}')
        print(f'sp coefs ratio: {np.mean(s_vals) / np.mean(s_vals_ood)}')
        
    if value_dict is None:
        value_dict = {}
        value_dict['c_vals'] = c_vals
        value_dict['c_vals_ood'] = c_vals_ood
        value_dict['s_vals'] = s_vals
        value_dict['s_vals_ood'] = s_vals_ood
        
        return value_dict

def get_axis(embeddings):
    
    core_ax1 = normalize(embeddings[f'1_{names[1]}'].mean(0, keepdims=False) - \
                         embeddings[f'0_{names[1]}'].mean(0, keepdims=False))
    core_ax2 = normalize(embeddings[f'1_{names[0]}'].mean(0, keepdims=False) - \
                         embeddings[f'0_{names[0]}'].mean(0, keepdims=False))
    core_ax = 0.5 * core_ax1 + 0.5 * core_ax2
    
    sp_ax1 = normalize(embeddings[f'1_{names[1]}'].mean(0, keepdims=False) - \
                         embeddings[f'1_{names[0]}'].mean(0, keepdims=False))
    sp_ax2 = normalize(embeddings[f'0_{names[1]}'].mean(0, keepdims=False) - \
                         embeddings[f'0_{names[0]}'].mean(0, keepdims=False))
    sp_ax = 0.5 * sp_ax1 + 0.5 * sp_ax2
    
    return core_ax, sp_ax


class MLP(nn.Module):
    def __init__(self, n_feat=n_feats, n_out=2):
        super().__init__()
        self.layer = nn.Linear(n_feat, n_out)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout(x)
        return self.layer(x)
    
core_ax, sp_ax = get_axis(train_dict)
_ = visualize_correlations(test_dict, core_ax, sp_ax)


mlp = MLP().to(device)  
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    
mlp.train()
for e in range(n_steps):
    
    data_np, lbl_np = prepare_train_data()
    
    data = torch.tensor(data_np, dtype=torch.float32).to(device)
    lbl = torch.tensor(lbl_np, dtype=torch.long).to(device) 
        
        
    optimizer.zero_grad()
    logits = mlp(data)
    weight = next(mlp.parameters())
    loss = loss_function(logits, lbl)
    loss.backward()
    optimizer.step()
    
    if e % int(n_steps / 4) == 0:
        mlp.eval()
        with torch.no_grad():
            
            g_names = []
            groups_acc = []
            
            for m in [0, 1]:
                for j, n in enumerate(names):
                    
                    name = f'{m}_{n}'
                    data_np = test_dict[name]
                    
                    lbl_np = m * np.ones(len(data_np))
                    
                    data_eval = torch.tensor(data_np, dtype=torch.float32).to(device)
                    lbl_eval = torch.tensor(lbl_np, dtype=torch.long).to(device)  
                    
                    pred = mlp(data_eval).max(-1)[1]
                    acc = (pred == lbl_eval).float().mean().item()
                    groups_acc.append(np.round(acc,4))
                    g_names.append(name + '_' + str(int(lbl_np[0])))
                    
            print(g_names)
            print('test acc:', groups_acc)
