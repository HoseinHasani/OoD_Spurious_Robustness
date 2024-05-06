import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import dist_utils
import warnings
warnings.filterwarnings("ignore")

normalize_embs = True
apply_mixup = False

batch_size = 64


n_steps = 160
n_feats = 2048
output_size = n_feats // 2


sp_rate = 0.90
alpha_refine = 0.99
alpha_ood = 0.6

lbl_scale = 0.1


backbones = ['dino', 'res50', 'res18']
backbone = backbones[1]
resnet_types = ['pretrained', 'finetuned', 'scratch']
resnet_type = resnet_types[0]

core_class_names = ['0', '1']
ood_class_names = ['0', '1']
sp_class_names = ['0', '1']
place_names = ['land', 'water']

data_path = '../embeddings/'



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if backbone == 'dino':
    in_data_embs0 = np.load(data_path + 'waterbird_embs_DINO.npy', allow_pickle=True).item()
elif backbone == 'res50':
    in_data_embs0 = np.load(data_path + f'wb_embs_{backbone}_{resnet_type}.npy', allow_pickle=True).item()

ood_embs0 = {}
if backbone == 'dino':
    dict_ = np.load(data_path + 'OOD_land_DINO_eval.npy', allow_pickle=True).item()
elif backbone == 'res50':
    dict_ = np.load(data_path + f'OOD_land_res50_eval.npy', allow_pickle=True).item()
    
ood_embs0['0'] = np.array([dict_[key].squeeze() for key in dict_.keys()])

if backbone == 'dino':
    dict_ = np.load(data_path + 'OOD_water_DINO_eval.npy', allow_pickle=True).item()
elif backbone == 'res50':
    dict_ = np.load(data_path + f'OOD_water_res50_eval.npy', allow_pickle=True).item()
ood_embs0['1'] = np.array([dict_[key].squeeze() for key in dict_.keys()])

grouped_embs0 = {}
grouped_embs_train0 = {}

for key in in_data_embs0.keys():
    emb = in_data_embs0[key].squeeze()
    label = key[0]
    place = key[2]
    split = key[4]
    name = f'{label}_{place}'
    
    if split != '0':
        if name not in grouped_embs0.keys():
            grouped_embs0[name] = []
        
        grouped_embs0[name].append(emb)
    else:
        if name not in grouped_embs_train0.keys():
            grouped_embs_train0[name] = []
        
        grouped_embs_train0[name].append(emb)

grouped_embs0 = {name: np.array(grouped_embs0[name]) for name in grouped_embs0.keys()}
grouped_embs_train0 = {name: np.array(grouped_embs_train0[name]) for name in grouped_embs_train0.keys()}



def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


grouped_embs = {name: grouped_embs0[name] for name in grouped_embs0.keys()}
grouped_embs_train = {name: grouped_embs_train0[name] for name in grouped_embs_train0.keys()}
ood_embs = {name: ood_embs0[name] for name in ood_embs0.keys()}


if normalize_embs:
    train_dict = {key: normalize(grouped_embs_train0[key]) for key in grouped_embs_train0.keys()}
    test_dict = {key: normalize(grouped_embs0[key]) for key in grouped_embs0.keys()}
    ood_dict = {key: normalize(ood_embs0[key]) for key in ood_embs0.keys()}
else:
    train_dict = grouped_embs_train0
    test_dict = grouped_embs
    ood_dict = ood_embs0



l_maj0 = len(train_dict[f'{core_class_names[0]}_{sp_class_names[0]}'])
l_min0 = len(train_dict[f'{core_class_names[0]}_{sp_class_names[1]}'])

l_maj1 = len(train_dict[f'{core_class_names[1]}_{sp_class_names[1]}'])
l_min1 = len(train_dict[f'{core_class_names[1]}_{sp_class_names[0]}'])


def sample_data(data_dict, n_data, sp_rate):
    
    n_maj = int(sp_rate * n_data)
    n_min = n_data - n_maj
    
    ind_0_maj = np.random.choice(l_maj0, size=n_maj, replace=False).ravel()
    ind_0_min = np.random.choice(l_min0, size=n_min, replace=False).ravel()

    ind_1_maj = np.random.choice(l_maj1, size=n_maj, replace=False).ravel()
    ind_1_min = np.random.choice(l_min1, size=n_min, replace=False).ravel()
        
    data0 = np.concatenate([data_dict[f'{core_class_names[0]}_{sp_class_names[0]}'][ind_0_maj],
                            data_dict[f'{core_class_names[0]}_{sp_class_names[1]}'][ind_0_min]])
    
    data1 = np.concatenate([data_dict[f'{core_class_names[1]}_{sp_class_names[1]}'][ind_1_maj],
                            data_dict[f'{core_class_names[1]}_{sp_class_names[0]}'][ind_1_min]])
    
    data_np1 = np.concatenate([
                            data0,
                            data1,
                            ])
    
    if apply_mixup:
        ind_0_maj = np.random.choice(l_maj0, size=n_maj, replace=False).ravel()
        ind_0_min = np.random.choice(l_min0, size=n_min, replace=False).ravel()
    
        ind_1_maj = np.random.choice(l_maj1, size=n_maj, replace=False).ravel()
        ind_1_min = np.random.choice(l_min1, size=n_min, replace=False).ravel()
            
        data0 = np.concatenate([data_dict[f'{core_class_names[0]}_{sp_class_names[0]}'][ind_0_maj],
                                data_dict[f'{core_class_names[0]}_{sp_class_names[1]}'][ind_0_min]])
        
        data1 = np.concatenate([data_dict[f'{core_class_names[1]}_{sp_class_names[1]}'][ind_1_maj],
                                data_dict[f'{core_class_names[1]}_{sp_class_names[0]}'][ind_1_min]])
        
        data_np2 = np.concatenate([
                                data0,
                                data1,
                                ])
        
        alpha_vals = np.random.rand(2 * n_data)[:, None]
        inds_ = np.random.choice(n_data, n_data//4, replace=False).ravel()
        
        alpha_vals[inds_] = 0
        
        data_np = alpha_vals * data_np1 + (1 - alpha_vals) * data_np2
    
    else:
        data_np = data_np1
        
    
    lbl0 = np.zeros((len(data0)))
    lbl1 = np.ones((len(data1)))
    
    lbl_np = np.concatenate([
                            lbl0,
                            lbl1,
                            ])
    
    return data_np, lbl_np


def sample_ood_data(n_data, sp_rate):
    
    n_maj = int(sp_rate * n_data)
    n_min = n_data - n_maj
    
    ind_0_maj = np.random.choice(l_maj0, size=n_maj, replace=False).ravel()
    ind_0_min = np.random.choice(l_min0, size=n_min, replace=False).ravel()

    ind_1_maj = np.random.choice(l_maj1, size=n_maj, replace=False).ravel()
    ind_1_min = np.random.choice(l_min1, size=n_min, replace=False).ravel()
        
    data0 = np.concatenate([pseudo_ood_dict[f'{core_class_names[0]}_{sp_class_names[0]}'][ind_0_maj],
                            pseudo_ood_dict[f'{core_class_names[0]}_{sp_class_names[1]}'][ind_0_min]])
    
    data1 = np.concatenate([pseudo_ood_dict[f'{core_class_names[1]}_{sp_class_names[1]}'][ind_1_maj],
                            pseudo_ood_dict[f'{core_class_names[1]}_{sp_class_names[0]}'][ind_1_min]])
    
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
        c_vals_ = np.abs(np.dot(normalize(embeddings[key]), core_ax))
        s_vals_ = np.abs(np.dot(normalize(embeddings[key]), sp_ax))
        
        c_vals.append(c_vals_)
        s_vals.append(s_vals_)

    c_vals_ood = []        
    s_vals_ood = []

    for key in ood_embeddings.keys():
        c_vals_ = np.abs(np.dot(normalize(ood_embeddings[key]), core_ax))
        s_vals_ = np.abs(np.dot(normalize(ood_embeddings[key]), sp_ax))
        
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
    
    core_ax1 = embeddings[f'{core_class_names[1]}_{sp_class_names[0]}'].mean(0, keepdims=False) - \
                         embeddings[f'{core_class_names[0]}_{sp_class_names[0]}'].mean(0, keepdims=False)
    core_ax2 = embeddings[f'{core_class_names[1]}_{sp_class_names[1]}'].mean(0, keepdims=False) - \
                         embeddings[f'{core_class_names[0]}_{sp_class_names[1]}'].mean(0, keepdims=False)
    core_ax = 0.5 * core_ax1 + 0.5 * core_ax2
    
    sp_ax1 = embeddings[f'{core_class_names[0]}_{sp_class_names[1]}'].mean(0, keepdims=False) - \
                         embeddings[f'{core_class_names[0]}_{sp_class_names[0]}'].mean(0, keepdims=False)
    sp_ax2 = embeddings[f'{core_class_names[1]}_{sp_class_names[1]}'].mean(0, keepdims=False) - \
                         embeddings[f'{core_class_names[1]}_{sp_class_names[0]}'].mean(0, keepdims=False)
    sp_ax = 0.5 * sp_ax1 + 0.5 * sp_ax2
    
    print('axis ratio:', np.linalg.norm(core_ax) / np.linalg.norm(sp_ax))
    
    core_ax_norm = 0.5 * normalize(core_ax1) + 0.5 * normalize(core_ax2)
    sp_ax_norm = 0.5 * normalize(sp_ax1) + 0.5 * normalize(sp_ax2)
    
    return core_ax, sp_ax, core_ax_norm, sp_ax_norm


    

def get_embeddings(model, group_data, max_l=32):
    
    model.eval()
    embeddings = {}
    logits = {}
    
    for key in group_data.keys():
        with torch.no_grad():
            data_np = group_data[key]
            data = torch.tensor(data_np, dtype=torch.float32).to(device)
            if len(data) > max_l:
                features = []
                logits_list = []
                for b in range(len(data) // max_l):
                    batch_data = data[b * max_l: (b + 1) * max_l]
                    feats, logs = model(batch_data.to(device))
                    
                    logits_list.append(logs)
                    features.append(feats)
                    
                features = torch.cat(features).cpu().numpy()
                logits_list = torch.cat(logits_list).cpu().numpy()
            else:
                features, logits_list = model(data.to(device)).cpu().numpy()
                
        embeddings[key] = features.squeeze()
        logits[key] = logits_list.squeeze()
        
    return embeddings, logits

def get_class_dicts(input_dict):
    class_dicts = []
    for core_name in core_class_names:
        class_dict = {}
        for sp_name in sp_class_names:
            name = f'{core_name}_{sp_name}'
            class_dict[name] = input_dict[name]
        class_dicts.append(class_dict)

    return class_dicts


def plot_dict_hist(dict_data, fig_name):
    data = []
    for name in dict_data:
        data.append(dict_data[name])
    
    data = np.concatenate(data)
    
    plt.hist(data, 100, histtype='step', linewidth=1.5, label=fig_name)
    plt.legend()


    
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
    
print()
dist_utils.calc_dists_ratio(train_dict, ood_dict)
dist_utils.calc_dists_ratio(test_dict, ood_dict)

train_dict_list = get_class_dicts(train_dict)
test_dict_list = get_class_dicts(test_dict)

ood_embs = np.concatenate([ood_dict[key] for key in ood_dict.keys()])
pseudo_ood_embs = np.concatenate([pseudo_ood_dict[key] for key in pseudo_ood_dict.keys()])

train_prototypes = [train_dict[key].mean(0) for key in train_dict.keys()]

print('OOD:')
#dist_utils.calc_ROC(test_dict_list[0], ood_embs, prototypes=train_dict_list[0])
#dist_utils.calc_ROC(test_dict_list[1], ood_embs, prototypes=train_dict_list[1])
dist_utils.calc_ROC(test_dict, ood_embs, prototypes=train_prototypes)

print('PSEUDO-OOD:')
#dist_utils.calc_ROC(test_dict_list[0], pseudo_ood_embs, prototypes=train_dict_list[0])
#dist_utils.calc_ROC(test_dict_list[1], pseudo_ood_embs, prototypes=train_dict_list[1])
dist_utils.calc_ROC(test_dict, pseudo_ood_embs, prototypes=train_prototypes)


# ood_embs = np.concatenate([pseudo_ood_dict[key] for key in pseudo_ood_dict.keys()])
# print()
# dist_utils.calc_ROC(train_dict_list[0], ood_embs)
# dist_utils.calc_ROC(train_dict_list[1], ood_embs)

core_ax_torch = torch.tensor(core_ax, dtype=torch.float32).to(device)
sp_ax_torch = torch.tensor(sp_ax, dtype=torch.float32).to(device)


class MLP(nn.Module):
    def __init__(self, n_feat=n_feats, n_out=2):
        super().__init__()
        self.layer1 = nn.Linear(n_feat, n_feat)
        self.layer2 = nn.Linear(n_feat, output_size)
        self.layer3 = nn.Linear(output_size, n_out)
        #self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
#        x = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x))
        x3 = self.layer3(x2)
        return x2, x3
    
    
mlp = MLP().to(device)  
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    

for e in range(n_steps):
    
    optimizer.zero_grad()
    
    data_np, lbl_np = sample_data(train_dict, batch_size, sp_rate=sp_rate)
            
    
    data = torch.tensor(data_np, dtype=torch.float32).to(device)
    lbl = torch.tensor(lbl_np, dtype=torch.long).to(device) 
        
    feats, logits = mlp(data)
    ce_loss = loss_function(logits, lbl)
    
        
    ce_loss.backward()
    optimizer.step()
    
    if e > 2 and e%4 == 0:
        mlp.eval()
        
        with torch.no_grad():
            
            g_names = []
            groups_acc = []
            
            for m in core_class_names:
                for n in core_class_names:
                    
                    name = f'{m}_{n}'
                    data_np = test_dict[name]
                    
                    lbl_np = int(m) * np.ones(len(data_np))
                    
                    data_eval = torch.tensor(data_np, dtype=torch.float32).to(device)
                    lbl_eval = torch.tensor(lbl_np, dtype=torch.long).to(device)  
                    
                    feats, logits = mlp(data_eval)
                    
                    pred = logits.max(-1)[1]
                    acc = (pred == lbl_eval).float().mean().item()
                    groups_acc.append(np.round(acc,4))
                    g_names.append(name)
                    
            print(g_names)
            print('test acc:', groups_acc)

        train_emb_dict, train_log_dict = get_embeddings(mlp, train_dict)
        test_emb_dict, test_log_dict = get_embeddings(mlp, test_dict)
        ood_emb_dict, ood_log_dict = get_embeddings(mlp, ood_dict)
        pseudo_ood_emb_dict, pseudo_ood_log_dict = get_embeddings(mlp, pseudo_ood_dict)
        
        
        ood_embs = np.concatenate([ood_emb_dict[key] for key in ood_emb_dict.keys()])
        ood_logits = np.concatenate([ood_log_dict[key] for key in ood_log_dict.keys()])
        
        pseudo_ood_embs = np.concatenate([pseudo_ood_emb_dict[key] for key in \
                                          pseudo_ood_emb_dict.keys()])     

        train_emb_prototypes = [train_emb_dict[key].mean(0) for key in train_emb_dict.keys()]
        
        print('OOD:')
        
        dist_utils.calc_ROC(test_emb_dict, ood_embs, prototypes=train_emb_prototypes)

        print('Probs (test):')
        dist_utils.calc_probs_ROC(test_log_dict, ood_logits)

        print('Probs (train):')
        dist_utils.calc_probs_ROC(train_log_dict, ood_logits)
                
                        
        
        
        
        print()
        dist_utils.calc_dists_ratio(train_emb_dict, ood_emb_dict)
        dist_utils.calc_dists_ratio(test_emb_dict, ood_emb_dict)

        core_ax, sp_ax, core_ax_norm, sp_ax_norm = get_axis(train_emb_dict)
        print('ax correlation: ', np.dot(core_ax_norm, sp_ax_norm))
        
        
        core_ax_torch = torch.tensor(core_ax, dtype=torch.float32).to(device)
        sp_ax_torch = torch.tensor(sp_ax, dtype=torch.float32).to(device)
        
        mlp.train()