import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import dist_utils
import os
import tqdm
import warnings
warnings.filterwarnings("ignore")

normalize_embs = True

n_steps = 1000
n_feats = 1024
batch_size = 128
sp_rate = 0.5

names = ['automobile', 'truck']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_path = 'data'
train_dict0 = np.load(f'{data_path}/Dominoes_train_embs.npy', allow_pickle=True).item()
test_dict0 = np.load(f'{data_path}/Dominoes_test_embs.npy', allow_pickle=True).item()
ood_dict0 = np.load(f'{data_path}/Dominoes_ood_embs.npy', allow_pickle=True).item()


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

if normalize_embs:
    train_dict = {key: normalize(train_dict0[key]) for key in train_dict0.keys()}
    test_dict = {key: normalize(test_dict0[key]) for key in test_dict0.keys()}
    ood_dict = {key: normalize(ood_dict0[key]) for key in ood_dict0.keys()}
else:
    train_dict = train_dict0
    test_dict = test_dict0
    ood_dict = ood_dict0



l_maj = len(train_dict[f'0_{names[0]}'])
l_min = len(train_dict[f'1_{names[0]}'])

def prepare_train_data():
    ind_0_maj = np.random.choice(l_maj, size=int(sp_rate * batch_size), replace=False).ravel()
    ind_0_min = np.random.choice(l_min, size=int((1 - sp_rate) * batch_size), replace=False).ravel()

    ind_1_maj = np.random.choice(l_maj, size=int(sp_rate * batch_size), replace=False).ravel()
    ind_1_min = np.random.choice(l_min, size=int((1 - sp_rate) * batch_size), replace=False).ravel()
        
    data_np = np.concatenate([
                            train_dict[f'0_{names[0]}'][ind_0_maj],
                            train_dict[f'1_{names[0]}'][ind_1_min],
                            train_dict[f'1_{names[1]}'][ind_1_maj],
                            train_dict[f'0_{names[1]}'][ind_0_min],
                            ])
    
    lbl_np = np.concatenate([
                            np.zeros(len(ind_0_maj)),
                            np.ones(len(ind_1_min)),
                            np.ones(len(ind_1_maj)),
                            np.zeros(len(ind_0_min)),
                            ])
    
    return data_np, lbl_np
    

def visualize_correlations(embeddings, ood_embeddings, core_ax, sp_ax,
                           value_dict=None, print_logs=False):
    
    c_vals = []
    s_vals = []
    
    for key in embeddings.keys():
        c_vals_ = np.abs(np.dot(embeddings[key], core_ax))
        s_vals_ = np.abs(np.dot(embeddings[key], sp_ax))
        
        c_vals.append(c_vals_)
        s_vals.append(s_vals_)

    c_vals_ood = []        
    s_vals_ood = []

    for key in ood_embeddings.keys():
        c_vals_ = np.abs(np.dot(ood_embeddings[key], core_ax))
        s_vals_ = np.abs(np.dot(ood_embeddings[key], sp_ax))
        
        c_vals_ood.append(c_vals_)
        s_vals_ood.append(s_vals_)

    c_vals = np.concatenate(c_vals)
    c_vals_ood = np.concatenate(c_vals_ood)
    s_vals = np.concatenate(s_vals)
    s_vals_ood = np.concatenate(s_vals_ood)

    
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.hist(c_vals, 25, histtype='step', density=False, linewidth=2.5, label='embs', color='tab:blue')
    plt.hist(c_vals_ood, 25, histtype='step', density=False, linewidth=2.5, label='ood', color='tab:orange')
    plt.title('core alignment')
    plt.subplot(122)
    plt.hist(s_vals, 25, histtype='step', density=False, linewidth=2.5, label='embs', color='tab:blue')
    plt.hist(s_vals_ood, 25, histtype='step', density=False, linewidth=2.5, label='ood', color='tab:orange')
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
    
    core_ax1 = embeddings[f'1_{names[1]}'].mean(0, keepdims=False) - \
                         embeddings[f'0_{names[1]}'].mean(0, keepdims=False)
    core_ax2 = embeddings[f'1_{names[0]}'].mean(0, keepdims=False) - \
                         embeddings[f'0_{names[0]}'].mean(0, keepdims=False)
    core_ax = 0.5 * core_ax1 + 0.5 * core_ax2
    
    sp_ax1 = embeddings[f'1_{names[1]}'].mean(0, keepdims=False) - \
                         embeddings[f'1_{names[0]}'].mean(0, keepdims=False)
    sp_ax2 = embeddings[f'0_{names[1]}'].mean(0, keepdims=False) - \
                         embeddings[f'0_{names[0]}'].mean(0, keepdims=False)
    sp_ax = 0.5 * sp_ax1 + 0.5 * sp_ax2
    
    print('axis ratio:', np.linalg.norm(core_ax) / np.linalg.norm(sp_ax))
    
    core_ax_norm = 0.5 * normalize(core_ax1) + 0.5 * normalize(core_ax2)
    sp_ax_norm = 0.5 * normalize(sp_ax1) + 0.5 * normalize(sp_ax2)
    
    return core_ax, sp_ax, core_ax_norm, sp_ax_norm


def alignment_score_v2(embs, core_ax, sp_ax, target, ood_embs=None,
                       alpha_l2=0.9, alpha_sp=1.9, alpha_ood=1.):
    
    alignment_func = torch.nn.CosineSimilarity(dim=-1)
    labels = torch.argmax(target, dim=-1)
    
    core_alignment = alignment_func(embs, core_ax) * (2 * labels - 1)
    avg_core_alignment = torch.abs(core_alignment).mean().detach().item()
    #core_alignment_clipped = torch.clip(core_alignment, -avg_core_alignment, avg_core_alignment)
    core_alignment_clipped = core_alignment
    
    sp_alignment = torch.abs(alignment_func(embs, sp_ax))
    avg_sp_alignment = torch.abs(sp_alignment).mean().detach().item()
    #sp_alignment_clipped = torch.clip(sp_alignment, avg_sp_alignment, 1.)
    sp_alignment_clipped = sp_alignment
    
    alignment = core_alignment_clipped.mean() - alpha_sp * sp_alignment_clipped.mean()
    
    
    if ood_embs is not None:
        ood_core_alignment = torch.abs(alignment_func(ood_embs, core_ax))
        avg_core_alignment = torch.abs(ood_core_alignment).mean().detach().item()
        ood_core_alignment_clipped = torch.clip(ood_core_alignment, avg_core_alignment, 1.)
    
        alignment -= ood_core_alignment_clipped.mean()
        
        print(ood_core_alignment_clipped.mean().item())
        
    l2_reg = torch.square(embs).mean()
    alignment -= alpha_l2 * l2_reg
    
    print(core_alignment_clipped.mean().item(), sp_alignment.mean().item(), l2_reg.item())
    
    return alignment

def get_embeddings(model, group_data, max_l=32):
    
    model.eval()
    embeddings = {}
    
    for key in group_data.keys():
        with torch.no_grad():
            data_np = group_data[key]
            data = torch.tensor(data_np, dtype=torch.float32).to(device)
            if len(data) > max_l:
                features = []
                for b in range(len(data) // max_l):
                    batch_data = data[b * max_l: (b + 1) * max_l]
                    feats, _ = model(batch_data.to(device)) 
                    features.append(feats)
                features = torch.cat(features).cpu().numpy()
            else:
                features, _ = model(data.to(device)).cpu().numpy()
        embeddings[key] = features.squeeze()

    return embeddings
    

class MLP(nn.Module):
    def __init__(self, n_feat=n_feats, n_out=2):
        super().__init__()
        self.layer1 = nn.Linear(n_feat, n_feat)
        self.layer2 = nn.Linear(n_feat, n_feat//1)
        self.layer3 = nn.Linear(n_feat//1, n_out)
        #self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        x3 = self.layer3(x2)
        return x2, x3
    
def get_class_dicts(input_dict):
    class_dicts = []
    for core_name in ['0', '1']:
        class_dict = {}
        for sp_name in names:
            name = f'{core_name}_{sp_name}'
            class_dict[name] = input_dict[name]
        class_dicts.append(class_dict)

    return class_dicts


core_ax, sp_ax, core_ax_norm, sp_ax_norm = get_axis(train_dict)
print('ax correlation: ', np.dot(core_ax, sp_ax))
_ = visualize_correlations(test_dict, ood_dict, core_ax, sp_ax)

dist_utils.calc_dists_ratio(train_dict, ood_dict)
dist_utils.calc_dists_ratio(test_dict, ood_dict)

train_dict_list = get_class_dicts(train_dict)
ood_embs = np.concatenate([ood_dict[key] for key in ood_dict.keys()])
print()
dist_utils.calc_ROC(train_dict_list[0], ood_embs)
dist_utils.calc_ROC(train_dict_list[1], ood_embs)


core_ax_torch = torch.tensor(core_ax, dtype=torch.float32).to(device)
sp_ax_torch = torch.tensor(sp_ax, dtype=torch.float32).to(device)

mlp = MLP().to(device)  
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    

for e in range(n_steps):
    
    optimizer.zero_grad()
    
    data_np, lbl_np = prepare_train_data()
    
    data = torch.tensor(data_np, dtype=torch.float32).to(device)
    lbl = torch.tensor(lbl_np, dtype=torch.long).to(device) 
        
    feats, logits = mlp(data)
    ce_loss = loss_function(logits, lbl)
    
    
    if e > 3:
        alignment_val = alignment_score_v2(feats, core_ax_torch, sp_ax_torch, lbl)
        loss = ce_loss - 0.0 * alignment_val
    else:
        loss = ce_loss
        
    loss.backward()
    optimizer.step()
    
    if e > 2:
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
                    
                    feats, logits = mlp(data_eval)
                    
                    pred = logits.max(-1)[1]
                    acc = (pred == lbl_eval).float().mean().item()
                    groups_acc.append(np.round(acc,4))
                    g_names.append(name + '_' + str(int(lbl_np[0])))
                    
            print(g_names)
            print('test acc:', groups_acc)

        train_emb_dict = get_embeddings(mlp, train_dict)
        test_emb_dict = get_embeddings(mlp, test_dict)
        ood_emb_dict = get_embeddings(mlp, ood_dict)

        train_dict_list = get_class_dicts(train_emb_dict)
        ood_embs = np.concatenate([ood_emb_dict[key] for key in ood_emb_dict.keys()])
        
        print()
        dist_utils.calc_ROC(train_dict_list[0], ood_embs)
        dist_utils.calc_ROC(train_dict_list[1], ood_embs)


        dist_utils.calc_dists_ratio(train_emb_dict, ood_emb_dict)
        dist_utils.calc_dists_ratio(test_emb_dict, ood_emb_dict)
        
        core_ax, sp_ax, core_ax_norm, sp_ax_norm = get_axis(train_emb_dict)
        print('ax correlation: ', np.dot(core_ax, sp_ax))
        
        
        core_ax_torch = torch.tensor(core_ax, dtype=torch.float32).to(device)
        sp_ax_torch = torch.tensor(sp_ax, dtype=torch.float32).to(device)
        
        if e % 5 == 0:
            _ = visualize_correlations(test_emb_dict, ood_emb_dict, core_ax, sp_ax)
        
        mlp.train()