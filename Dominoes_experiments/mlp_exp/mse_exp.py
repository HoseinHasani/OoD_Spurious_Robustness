import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import dist_utils
import nn_utils
import os
import tqdm
import warnings
warnings.filterwarnings("ignore")

normalize_embs = True

batch_size = 128

alpha_refine = 0.5

n_steps = 100
n_feats = 1024
sp_rate = 0.9

lbl_scale = 0.1
output_size = n_feats // 10


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

def sample_data(n_data, sp_rate):
    
    n_maj = int(sp_rate * n_data)
    n_min = n_data - n_maj
    
    ind_0_maj = np.random.choice(l_maj, size=n_maj, replace=False).ravel()
    ind_0_min = np.random.choice(l_min, size=n_min, replace=False).ravel()

    ind_1_maj = np.random.choice(l_maj, size=n_maj, replace=False).ravel()
    ind_1_min = np.random.choice(l_min, size=n_min, replace=False).ravel()
        
    data0 = np.concatenate([train_dict[f'0_{names[0]}'][ind_0_maj],
                            train_dict[f'0_{names[1]}'][ind_0_min]])
    
    data1 = np.concatenate([train_dict[f'1_{names[1]}'][ind_1_maj],
                            train_dict[f'1_{names[0]}'][ind_1_min]])
    
    data_np = np.concatenate([
                            data0,
                            data1,
                            ])
    
    lbl0 = -lbl_scale * np.ones((len(data0), output_size))
    lbl1 = lbl_scale * np.ones((len(data1), output_size))
    
    lbl_np = np.concatenate([
                            lbl0,
                            lbl1,
                            ])
    
    return data_np, lbl_np

def sample_ood_data(n_data, sp_rate):
    
    n_maj = int(sp_rate * n_data)
    n_min = n_data - n_maj
    
    ind_0_maj = np.random.choice(l_maj, size=n_maj, replace=False).ravel()
    ind_0_min = np.random.choice(l_min, size=n_min, replace=False).ravel()

    ind_1_maj = np.random.choice(l_maj, size=n_maj, replace=False).ravel()
    ind_1_min = np.random.choice(l_min, size=n_min, replace=False).ravel()
        
    data0 = np.concatenate([pseudo_ood_dict[f'0_{names[0]}'][ind_0_maj],
                            pseudo_ood_dict[f'0_{names[1]}'][ind_0_min]])
    
    data1 = np.concatenate([pseudo_ood_dict[f'1_{names[1]}'][ind_1_maj],
                            pseudo_ood_dict[f'1_{names[0]}'][ind_1_min]])
    
    data_np = np.concatenate([
                            data0[None],
                            data1[None],
                            ])
    
    return data_np
    

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
                    feats = model(batch_data.to(device)) 
                    features.append(feats)
                features = torch.cat(features).cpu().numpy()
            else:
                features = model(data.to(device)).cpu().numpy()
        embeddings[key] = features.squeeze()

    return embeddings
    


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

pseudo_ood_dict = {}
for name in train_dict.keys():
    embs = train_dict[name]
    cr_coefs = np.dot(embs, core_ax_norm)
    refined = embs - alpha_refine * cr_coefs[:, None] * np.repeat(core_ax_norm[None], embs.shape[0], axis=0)
    pseudo_ood_dict[name] = refined

_ = visualize_correlations(pseudo_ood_dict, ood_dict, core_ax_norm, sp_ax_norm)


dist_utils.calc_dists_ratio(train_dict, ood_dict)
dist_utils.calc_dists_ratio(test_dict, ood_dict)

test_dict_list = get_class_dicts(test_dict)
ood_embs = np.concatenate([ood_dict[key] for key in ood_dict.keys()])
print()
dist_utils.calc_ROC(test_dict_list[0], ood_embs)
dist_utils.calc_ROC(test_dict_list[1], ood_embs)

# ood_embs = np.concatenate([pseudo_ood_dict[key] for key in pseudo_ood_dict.keys()])
# print()
# dist_utils.calc_ROC(train_dict_list[0], ood_embs)
# dist_utils.calc_ROC(train_dict_list[1], ood_embs)

core_ax_torch = torch.tensor(core_ax, dtype=torch.float32).to(device)
sp_ax_torch = torch.tensor(sp_ax, dtype=torch.float32).to(device)

mlp = nn_utils.MLP(n_feats,
                   n_outputs=output_size,
                   input_scale=np.sqrt(n_feats),
                   sigmoid_output=True).to(device) 
 
model = nn_utils.ProtoNet(mlp, device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
for e in range(n_steps):
    
    optimizer.zero_grad()
    
    data_np, lbl_np = sample_data(batch_size, sp_rate=sp_rate)
    
    
    data = torch.from_numpy(data_np).float().to(device)
    lbl = torch.tensor(lbl_np).float().to(device) 
    
        
    feats = mlp(data)
    mse_loss = loss_function(feats, lbl)
        
    mse_loss.backward()
    optimizer.step()
    
    if e > 2:
        mlp.eval()
        
        with torch.no_grad():
            
            data_np, lbl_np = sample_data(batch_size, sp_rate=sp_rate)
            
            
            data = torch.from_numpy(data_np).float().to(device)
            lbl = torch.tensor(lbl_np).float().to(device) 
            
                
            feats = mlp(data)
            mse_loss = loss_function(feats, lbl)
            
            print('Train MSE loss: ', mse_loss.item())

            
        train_emb_dict = get_embeddings(mlp, train_dict)
        test_emb_dict = get_embeddings(mlp, test_dict)
        ood_emb_dict = get_embeddings(mlp, ood_dict)
        
        test_dict_list = get_class_dicts(test_emb_dict)
        ood_embs = np.concatenate([ood_emb_dict[key] for key in ood_emb_dict.keys()])
        dist_utils.calc_ROC(test_dict_list[0], ood_embs)
        dist_utils.calc_ROC(test_dict_list[1], ood_embs)

        print()
        dist_utils.calc_dists_ratio(train_emb_dict, ood_emb_dict)
        dist_utils.calc_dists_ratio(test_emb_dict, ood_emb_dict)
        
        core_ax, sp_ax, core_ax_norm, sp_ax_norm = get_axis(train_emb_dict)
        print('ax correlation: ', np.dot(core_ax_norm, sp_ax_norm))
        
        
        core_ax_torch = torch.tensor(core_ax, dtype=torch.float32).to(device)
        sp_ax_torch = torch.tensor(sp_ax, dtype=torch.float32).to(device)
        
        if e % 5 == 0:
            _ = visualize_correlations(test_emb_dict, ood_emb_dict, core_ax_norm, sp_ax_norm)
        
        mlp.train()