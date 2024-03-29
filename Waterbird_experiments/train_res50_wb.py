from torchvision.models.resnet import Bottleneck, ResNet
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import argparse
from torchvision import transforms
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import glob

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet50(pretrained=True)
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, 2)

    def forward(self, x, return_feature=False, return_feature_list=False):
        features = self.model.conv1(x)
        features = self.model.bn1(features)
        features = self.model.relu(features)
        features = self.model.maxpool(features)

        features1 = self.model.layer1(features)
        features2 = self.model.layer2(features1)
        features3 = self.model.layer3(features2)
        features4 = self.model.layer4(features3)

        global_avg_pool = self.model.avgpool(features4)
        global_avg_pool = global_avg_pool.view(global_avg_pool.size(0), -1)

        output = self.model.fc(global_avg_pool)

        if return_feature:
            return output, global_avg_pool
        elif return_feature_list:
            return output, [features1, features2, features3, features4, global_avg_pool]
        else:
            return output

    def forward_threshold(self, x, threshold):
        features = self.model.conv1(x)
        features = self.model.bn1(features)
        features = self.model.relu(features)
        features = self.model.maxpool(features)

        features1 = self.model.layer1(features)
        features2 = self.model.layer2(features1)
        features3 = self.model.layer3(features2)
        features4 = self.model.layer4(features3)

        global_avg_pool = self.model.avgpool(features4)
        global_avg_pool = global_avg_pool.view(global_avg_pool.size(0), -1)

        output = self.model.fc(global_avg_pool)

        # Applying threshold to the output
        output = F.threshold(output, threshold, 0)

        return output

    def intermediate_forward(self, x, layer_index):
        # Choose the appropriate layer based on the layer_index
        if layer_index == 1:
            return self.model.layer1(x)
        elif layer_index == 2:
            return self.model.layer2(self.model.layer1(x))
        elif layer_index == 3:
            return self.model.layer3(self.model.layer2(self.model.layer1(x)))
        elif layer_index == 4:
            return self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(x))))
        else:
            raise ValueError("Invalid layer_index. Supported values are 1, 2, 3, and 4.")

    def get_fc(self):
        fc = self.model.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.fc



# Load pre-trained ResNet-50 model
custom_model = ResNet50()



class WaterbirdDataset(Dataset):
    def __init__(self, split, path, transform):
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        self.env_dict = {
            (0, 0): torch.Tensor(np.array([1,0,0,0])),
            (0, 1): torch.Tensor(np.array([0,1,0,0])),
            (1, 0): torch.Tensor(np.array([0,0,1,0])),
            (1, 1): torch.Tensor(np.array([0,0,0,1]))
        }
        self.split = split
        self.dataset_dir= path
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'metadata.csv'))
        self.metadata_df = self.metadata_df[self.metadata_df['split']==self.split_dict[self.split]]

        y_array = torch.Tensor(np.array(self.metadata_df['y'].values)).type(torch.LongTensor)
        self.y_array = self.metadata_df['y'].values

        self.place_array = self.metadata_df['place'].values
        self.filename_array = self.metadata_df['img_filename'].values
        self.transform = transform

        self.y_one_hot = nn.functional.one_hot(y_array, num_classes=2).type(torch.FloatTensor)

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        place = self.place_array[idx]
        img_filename = os.path.join(
            self.dataset_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)

        label = self.y_one_hot[idx]

        return img, label, self.env_dict[(y, place)]

    def get_raw_image(self,idx):
      scale = 256.0/224.0
      target_resolution = [224, 224]
      img_filename = os.path.join(
            self.dataset_dir,
            self.filename_array[idx])
      img = Image.open(img_filename).convert('RGB')
      transform = transforms.Compose([
          transforms.Resize(
              (int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
          transforms.CenterCrop(target_resolution),
          transforms.ToTensor(),
      ])
      return transform(img)



def sample_group_batch(args, n_g=32):
    
    metadata = pd.read_csv(os.path.join(args.data_path, 'metadata.csv'))
    
    file_names = metadata['img_filename'].tolist()
    labels = metadata['y'].tolist()
    splits = metadata['split'].tolist()
    places = metadata['place'].tolist()
    
    train_inds = np.argwhere(np.array(splits) == 0).ravel()
    
    group_sample_names = {}
    for label in set(labels):
        for place in set(places):
            inds_l = np.argwhere(np.array(labels) == label).ravel()
            inds_p = np.argwhere(np.array(places) == place).ravel()
            
            inds = set(inds_l).intersection(set(inds_p)).intersection(set(train_inds))
            inds = np.array(list(inds))
            selected_inds = np.random.choice(inds, n_g, replace=False).ravel()
            group_sample_names[f'{label}_{place}'] = np.array(file_names)[selected_inds]
    
    f_list = np.array(glob.glob(args.ood_data_path + '/*.jpg'))
    selected_inds = np.random.choice(len(f_list), 5 * n_g, replace=False).ravel()
    
    for k in range(5):
        group_sample_names[f'OOD_{k}'] = f_list[selected_inds[k * n_g: (k + 1) * n_g]]
    
    
    return group_sample_names
            

def load_group_batch(args, transform, device):
    
    group_sample_names = sample_group_batch(args)

    group_data_batch = {}
    
    for name in group_sample_names.keys():
        data = []
        for j in range(len(group_sample_names[name])):
            if 'OOD' in name:
                img_filename = group_sample_names[name][j]
            else:
                img_filename = os.path.join(
                    args.data_path,
                    group_sample_names[name][j])
            
            img = Image.open(img_filename).convert('RGB')
            img = transform(img)
            data.append(img)
        
        group_data_batch[name] = torch.stack(data).to(device).float()

    return group_data_batch

def get_waterbird_dataloader(split, transform, path, batch_size):
    kwargs = {'pin_memory': True, 'num_workers': 2, 'drop_last': False}
    dataset = WaterbirdDataset( split=split, path = path, transform = transform)
    if not split == 'train':

      dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, **kwargs)
    else:
      dataloader = DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True, **kwargs)
    return dataloader


def get_waterbird_loaders(path, batch_size):
    t_train = get_transform_cub(True)
    t_tst = get_transform_cub(False)
    trainloader = get_waterbird_dataloader('train', t_train, path, batch_size)
    valloader = get_waterbird_dataloader('val', t_tst, path, batch_size)
    testloader = get_waterbird_dataloader('test', t_tst, path, batch_size)

    return trainloader, valloader, testloader


def get_waterbird_dataset(split, path, transform):
    dataset = WaterbirdDataset(split=split, path = path, transform = transform)
    return dataset


def get_transform_cub(train):
    scale = 256.0/224.0
    target_resolution = [224, 224]
    assert target_resolution is not None

    if (not train):

      transform = transforms.Compose([
                transforms.Resize(
                    (int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                0.229, 0.224, 0.225]),
            ])

    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                            0.229, 0.224, 0.225]),
        ])

    return transform




def test_model(model, device, test_loader, set_name="test set"):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, env in test_loader:
            data, target = data.to(device).float(), target.to(device).float()
            output = model(data)

            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = torch.argmax(output, dim=1)
            label = torch.argmax(target, dim=1)
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f'\nPerformance on {set_name}: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)})\n')
    return 100. * correct / len(test_loader.dataset)

def alignment_score(embs, core_ax, sp_ax, target, alpha_sp=0.9):
    
    alignment_func = torch.nn.CosineSimilarity(dim=-1)
    labels = torch.argmax(target, dim=-1)
    
    core_alignment = alignment_func(embs, core_ax) * (2 * labels - 1)
    avg_core_alignment = torch.abs(core_alignment).mean().detach().item()
    core_alignment_clipped = torch.clip(core_alignment, -avg_core_alignment, avg_core_alignment)
    
    sp_alignment = torch.abs(alignment_func(embs, sp_ax))
    avg_sp_alignment = torch.abs(sp_alignment).mean().detach().item()
    sp_alignment_clipped = torch.clip(sp_alignment, avg_sp_alignment, 1.)
    
    alignment = core_alignment_clipped.mean() - alpha_sp * sp_alignment_clipped.mean()
    #print(torch.abs(core_alignment).mean().item(), sp_alignment.mean().item(), alignment.item())
    return alignment

def erm_train(model, device, train_loader, optimizer, epoch, group_data_batch, alpha=0.9):

    print('Extract group embeddings ...')
    embeddings, core_ax, sp_ax = get_axis(model, group_data_batch)
    visualize_correlations(embeddings, core_ax, sp_ax)
    
    criterion = nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target, env) in enumerate(tqdm(train_loader)):
        data, target = data.to(device).float(), target.to(device).float()
        optimizer.zero_grad()
        output, features = model(data, return_feature=True)
        alignment_val = alignment_score(features, core_ax, sp_ax, target)
        loss = criterion(output, target) - alpha * alignment_val
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 9:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader)}%)]\tLoss: {loss.item()}')

            embeddings, core_ax, sp_ax = get_axis(model, group_data_batch)
            visualize_correlations(embeddings, core_ax, sp_ax)
            model.train()
            
            
def visualize_correlations(embeddings, core_ax, sp_ax, print_logs=True):
    
    c_vals = []
    c_vals_ood = []

    s_vals = []
    s_vals_ood = []
    
    for key in embeddings.keys():
        c_vals_ = np.abs(np.dot(embeddings[key].cpu().numpy(),
                                            core_ax.cpu().numpy().squeeze()))
        s_vals_ = np.abs(np.dot(embeddings[key].cpu().numpy(),
                                            sp_ax.cpu().numpy().squeeze()))
        
        if 'OOD' in key:
            c_vals_ood.append(c_vals_)
            s_vals_ood.append(s_vals_)
        else:
            c_vals.append(c_vals_)
            s_vals.append(s_vals_)

    c_vals = np.concatenate(c_vals)
    c_vals_ood = np.concatenate(c_vals_ood)
    s_vals = np.concatenate(s_vals)
    s_vals_ood = np.concatenate(s_vals_ood)

    
    plt.figure()
    plt.hist(c_vals, 25, histtype='step', density=True, linewidth=2.5, label='embs')
    plt.hist(c_vals_ood, 25, histtype='step', density=True, linewidth=2.5, label='ood')
    plt.title('core alignment')
    plt.legend()
    
    plt.figure()
    plt.hist(s_vals, 25, histtype='step', density=True, linewidth=2.5, label='embs')
    plt.hist(s_vals_ood, 25, histtype='step', density=True, linewidth=2.5, label='ood')
    plt.title('sp alignment')
    plt.legend()
    
    if print_logs:
        print(f'core coefs ratio: {np.mean(c_vals) / np.mean(c_vals_ood)}')
        print(f'sp coefs ratio: {np.mean(s_vals) / np.mean(s_vals_ood)}')
    
    
def get_axis(model, group_data):
    
    model.eval()
    
    embeddings = {}
    for key in group_data.keys():
        with torch.no_grad():
            _, features = model(group_data[key], return_feature=True)
        embeddings[key] = features.squeeze()
    
    core_ax1 = F.normalize(embeddings['1_1'].mean(0, keepdims=True) - embeddings['0_1'].mean(0, keepdims=True))
    core_ax2 = F.normalize(embeddings['1_0'].mean(0, keepdims=True) - embeddings['0_0'].mean(0, keepdims=True))
    core_ax = 0.5 * core_ax1 + 0.5 * core_ax2
    
    sp_ax1 = F.normalize(embeddings['1_1'].mean(0, keepdims=True) - embeddings['1_0'].mean(0, keepdims=True))
    sp_ax2 = F.normalize(embeddings['0_1'].mean(0, keepdims=True) - embeddings['0_0'].mean(0, keepdims=True))
    sp_ax = 0.5 * sp_ax1 + 0.5 * sp_ax2
    
    return embeddings, core_ax, sp_ax

    
def train_and_test_erm(args):
    print("ERM...\n")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    all_train_loader, val_loader, test_loader = get_waterbird_loaders(path=args.data_path,
                                                           batch_size=args.batch_size)
    print('Load group samples ...')
    group_data_batch = load_group_batch(args, get_transform_cub(False), device)
    
    model = custom_model.to(device)
    # model.load_state_dict(torch.load('/home/user01/models/pretrained_ResNet50.model'))
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3, momentum=0.9)
    
    
    train_acc = []
    val_acc = []
    test_acc = []
    best_acc = 0
    print('Start training ...')
    for epoch in range(1, args.epoch_size):
        erm_train(model, device, all_train_loader, optimizer, epoch, group_data_batch)
        #train_acc.append(test_model(model, device, all_train_loader, set_name=f'train set epoch {epoch}'))
        val_acc.append(test_model(model, device, val_loader, set_name=f'validation set epoch {epoch}'))
        if val_acc[-1] > best_acc:
            best_acc = val_acc[-1]
            torch.save(model.state_dict(), os.path.join(args['ckpt_path'],
                                                        'resnet50_waterbirds_'+ str(args.r)+'_best_checkpoint_seed' + str(
                                                            args.seed) +  '_scratch.model'))

        test_acc.append(test_model(model, device, test_loader, set_name=f'test set epoch {epoch}'))
        print(f'acc: {np.mean(val_acc)}, {np.mean(test_acc)}')
    torch.save(model.state_dict(), os.path.join(args['ckpt_path'],
                                                        'resnet50_waterbirds_'+ str(args.r)+'_best_checkpoint_seed' + str(
                                                            args.seed) +  '_scratch.model'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_data_path = 'waterbird'
    parser.add_argument("--data_path", type=str, default=default_data_path, help="data path")
    parser.add_argument("--ckpt_path", type=str, default='resnet50_exps', help="checkpoint path")
    parser.add_argument("--dataset", type=str, default='Waterbirds')
    parser.add_argument("--ood_data_path", type=str, default='OOD_Datasets/placesbg/water')
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--backbone_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epoch_size", type=int, default=20)
    parser.add_argument("--sampling_mode", type=str, default='top-k')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--r", type=int, default=95)

    args, unknown = parser.parse_known_args()
    args = (args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    
    train_and_test_erm(args)