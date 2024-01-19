    import torch
    from torch import nn
    import torch.nn.functional as F
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import tqdm
    
    
    
    # set seed
    seed = 8
    n_feats = 1024
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
    
    
    mlp_sp = MLP().to(device)  
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp_sp.parameters(), lr=1e-3)
        
    print('Train spurious feature classifier:')
    mlp_sp.train()
        
    for e in tqdm.tqdm(range(500)):
        
        data_list = []
        for name in ['woman_black', 'woman_blond']:
            g_inds = np.random.choice(len(grouped_embs[name]), g_batch_size, replace=False)
            data_list.append(grouped_embs[name][g_inds])
        
        data0 = torch.tensor(np.concatenate(data_list), dtype=torch.float32).to(device)  
        lbl0 = torch.zeros(2 * g_batch_size, dtype=torch.long).to(device)  
        
        data_list = []
        for name in ['man_black', 'man_blond']:
            g_inds = np.random.choice(len(grouped_embs[name]), g_batch_size, replace=False)
            data_list.append(grouped_embs[name][g_inds])
        
        data1 = torch.tensor(np.concatenate(data_list), dtype=torch.float32).to(device)  
        lbl1 = torch.ones(2 * g_batch_size, dtype=torch.long).to(device)  
        
        data = torch.cat([data0, data1])
        lbl = torch.cat([lbl0, lbl1])
        
        optimizer.zero_grad()
        logits = mlp_sp(data)
        weight = next(mlp_sp.parameters())
        l1_loss = torch.abs(weight).sum()
        loss = loss_function(logits, lbl) + gamma * l1_loss
        loss.backward()
        optimizer.step()
        
        if e % 100 == 0:
            with torch.no_grad():
                plt.figure()
                weight = next(mlp_sp.parameters()).detach().cpu().numpy()
                plt.hist(np.abs(weight.ravel()), 100)
                plt.title(str(e))
                
                data00 = torch.tensor(np.concatenate([grouped_embs['woman_black'][:1000],
                                                     grouped_embs['woman_blond'][:1000]]),
                                                    dtype=torch.float32).to(device)
                lbl00 = torch.zeros(2000, dtype=torch.long).to(device) 
                
                data11 = torch.tensor(np.concatenate([grouped_embs['man_black'][:1000],
                                                     grouped_embs['man_blond'][:1000]]),
                                                    dtype=torch.float32).to(device)
                lbl11 = torch.ones(2000, dtype=torch.long).to(device) 
                data_ = torch.cat([data00, data11])
                lbl_ = torch.cat([lbl00, lbl11])
                pred = mlp_sp(data_).max(-1)[1]
                acc = (pred == lbl_).float().mean().item()
                print('train acc:', acc)
                
                
    weight = next(mlp_sp.parameters()).detach().cpu().numpy()
    n_feat = 100
    woman_feats = np.argsort(weight[0])[-n_feat:]
    man_feats = np.argsort(weight[1])[-n_feat:]
    negative_feats = {'woman': woman_feats, 'man': man_feats}
    merged_sp_feats = list(set(np.concatenate([woman_feats, man_feats])))
    
    ####################################################
        
    non_sp_feats = np.delete(np.arange(n_feats), merged_sp_feats)
    mlp = MLP(n_feat=len(non_sp_feats)).to(device)  
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        
    print('Train core feature classifier:')
    mlp.train()
        
    for e in tqdm.tqdm(range(500)):
        
        data_list = []
        for name in ['man_blond', 'woman_blond']:
            g_inds = np.random.choice(len(grouped_embs[name]), g_batch_size, replace=False)
            data_list.append(grouped_embs[name][g_inds])
        
        data0 = torch.tensor(np.concatenate(data_list), dtype=torch.float32).to(device)  
        lbl0 = torch.zeros(2 * g_batch_size, dtype=torch.long).to(device)  
        
        data_list = []
        for name in ['man_black', 'woman_black']:
            g_inds = np.random.choice(len(grouped_embs[name]), g_batch_size, replace=False)
            data_list.append(grouped_embs[name][g_inds])
        
        lbl1 = torch.ones(2 * g_batch_size, dtype=torch.long).to(device)  
        
        data = torch.cat([data0, data1])[:, non_sp_feats]
        lbl = torch.cat([lbl0, lbl1])
        
        optimizer.zero_grad()
        logits = mlp(data)
        weight = next(mlp.parameters())
        l1_loss = torch.abs(weight).sum()
        loss = loss_function(logits, lbl) + gamma * l1_loss
        loss.backward()
        optimizer.step()
        
        if e % 100 == 0:
            with torch.no_grad():
                plt.figure()
                weight = next(mlp.parameters()).detach().cpu().numpy()
                plt.hist(np.abs(weight.ravel()), 100)
                plt.title(str(e))
                
                data00 = torch.tensor(np.concatenate([grouped_embs['man_blond'][:1000],
                                                     grouped_embs['woman_blond'][:1000]]),
                                                    dtype=torch.float32).to(device)
                lbl00 = torch.zeros(2000, dtype=torch.long).to(device) 
                
                data11 = torch.tensor(np.concatenate([grouped_embs['man_black'][:1000],
                                                     grouped_embs['woman_black'][:1000]]),
                                                    dtype=torch.float32).to(device)
                lbl11 = torch.ones(2000, dtype=torch.long).to(device) 
                data_ = torch.cat([data00, data11])[:, non_sp_feats]
                lbl_ = torch.cat([lbl00, lbl11])
                pred = mlp(data_).max(-1)[1]
                acc = (pred == lbl_).float().mean().item()
                print('train acc:', acc)
                
                
    weight = next(mlp.parameters()).detach().cpu().numpy()
    n_feat = 100
    blond_feats = np.argsort(weight[0])[-n_feat:]
    black_feats = np.argsort(weight[1])[-n_feat:]
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
        
    from sklearn.linear_model import LogisticRegression
    
    merged_core_feats = list(set(np.concatenate([blond_feats, black_feats])))
    
    
    
    data0 = np.concatenate([grouped_embs['man_blond'][:1000], grouped_embs['woman_blond'][:1000]])
    lbl0 = np.zeros(2000)
    data1 = np.concatenate([grouped_embs['man_black'][:1000], grouped_embs['woman_black'][:1000]])
    lbl1 = np.ones(2000)
    
    x_train = np.concatenate([data0, data1])[:, non_sp_feats][:, merged_core_feats]
    y_train = np.concatenate([lbl0, lbl1])
    
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    
    data0 = np.concatenate([grouped_embs['man_blond'][-200:], grouped_embs['woman_blond'][-200:]])
    lbl0 = np.zeros(400)
    data1 = np.concatenate([grouped_embs['man_black'][-200:], grouped_embs['woman_black'][-200:]])
    lbl1 = np.ones(400)
    
    x_eval = np.concatenate([data0, data1])[:, non_sp_feats][:, merged_core_feats]
    y_eval = np.concatenate([lbl0, lbl1])
    
    preds = clf.predict(x_eval)
    eval_acc = 100 * (preds == y_eval).mean()
    print('BLOND / BLACK ACC:', eval_acc)
    
    data0 = np.concatenate([grouped_embs['woman_black'][:1000], grouped_embs['woman_blond'][:1000]])
    lbl0 = np.zeros(2000)
    data1 = np.concatenate([grouped_embs['man_black'][:1000], grouped_embs['man_blond'][:1000]])
    lbl1 = np.ones(2000)
    
    x_train = np.concatenate([data0, data1])[:, non_sp_feats][:, merged_core_feats]
    y_train = np.concatenate([lbl0, lbl1])
    
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    
    data0 = np.concatenate([grouped_embs['woman_black'][-200:], grouped_embs['woman_blond'][-200:]])
    lbl0 = np.zeros(400)
    data1 = np.concatenate([grouped_embs['man_black'][-200:], grouped_embs['man_blond'][-200:]])
    lbl1 = np.ones(400)
    
    x_eval = np.concatenate([data0, data1])[:, non_sp_feats][:, merged_core_feats]
    y_eval = np.concatenate([lbl0, lbl1])
    
    preds = clf.predict(x_eval)
    eval_acc = 100 * (preds == y_eval).mean()
    print('MAN / WOMAN ACC:', eval_acc)
    
    ####################################################