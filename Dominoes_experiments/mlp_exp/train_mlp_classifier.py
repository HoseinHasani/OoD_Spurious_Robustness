import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
import warnings
warnings.filterwarnings("ignore")

n_steps = 100
n_feats = 1024
batch_size = 128
sp_rate = 0.95


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_path = 'data'
train_dict0 = np.load(f'{data_path}/Dominoes_train_embs.npy', allow_pickle=True).item()
test_dict0 = np.load(f'{data_path}/Dominoes_test_embs.npy', allow_pickle=True).item()

def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

train_dict = {key: normalize(train_dict0[key]) for key in train_dict0.keys()}
test_dict = {key: normalize(test_dict0[key]) for key in test_dict0.keys()}


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
    


class MLP(nn.Module):
    def __init__(self, n_feat=n_feats, n_out=2):
        super().__init__()
        self.layer = nn.Linear(n_feat, n_out)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout(x)
        return self.layer(x)
    

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
