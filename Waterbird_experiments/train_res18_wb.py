from torchvision.models.resnet import Bottleneck, ResNet
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import math
import argparse
import copy
from torchvision import transforms
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import math
import argparse
import copy
from torchvision import transforms
import sys


import torch.nn as nn
import torch.nn.functional as F
import torchvision



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)  
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

        global_avg_pool = F.adaptive_avg_pool2d(features4, (1, 1))
        global_avg_pool = global_avg_pool.view(global_avg_pool.size(0), -1)

        output = self.model.fc(global_avg_pool)

        if return_feature:
            return output, global_avg_pool
        elif return_feature_list:
            return output, [features1, features2, features3, features4, global_avg_pool]
        else:
            return output

    def forward_threshold(self, x, threshold):
        output = self.forward(x)
        # Applying threshold to the output
        output = F.threshold(output, threshold, 0)
        return output

    def intermediate_forward(self, x, layer_index):
        out = self.model.relu(self.model.bn1(self.model.conv1(x)))
        out = self.model.maxpool(out)

        out = self.model.layer1(out)
        if layer_index == 1:
            return out

        out = self.model.layer2(out)
        if layer_index == 2:
            return out

        out = self.model.layer3(out)
        if layer_index == 3:
            return out

        out = self.model.layer4(out)
        if layer_index == 4:
            return out

        raise ValueError("Invalid layer_index. Supported values are 1, 2, 3, and 4.")

    def get_fc(self):
        fc = self.model.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.fc

custom_model = ResNet18()


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


def erm_train(model, device, train_loader, optimizer, epoch):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target, env) in enumerate(tqdm(train_loader)):
        data, target = data.to(device).float(), target.to(device).float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader)}%)]\tLoss: {loss.item()}')


def train_and_test_erm(args):
    print("ERM...\n")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    all_train_loader, val_loader, test_loader = get_waterbird_loaders(path=args.data_path,
                                                           batch_size=args.batch_size)

    model = custom_model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3, momentum=0.9)

    train_acc = []
    val_acc = []
    test_acc = []
    best_acc = 0
    for epoch in range(1, 100):
        erm_train(model, device, all_train_loader, optimizer, epoch)
        train_acc.append(test_model(model, device, all_train_loader, set_name=f'train set epoch {epoch}'))
        val_acc.append(test_model(model, device, val_loader, set_name=f'validation set epoch {epoch}'))
        if val_acc[-1] > best_acc:
            best_acc = val_acc[-1]
            torch.save(model.state_dict(), os.path.join('/home/user01/OpenOODv1.5/checkpoints/waterbirds/resnet18',
                                                        'resnet18_waterbirds_'+ str(args.r)+'_best_checkpoint_seed' + str(
                                                            args.seed) +  '.model'))

        test_acc.append(test_model(model, device, test_loader, set_name=f'test set epoch {epoch}'))

    torch.save(model.state_dict(), os.path.join('/home/user01/OpenOODv1.5/checkpoints/waterbirds/resnet18',
                                                        'resnet18_waterbirds_'+ str(args.r)+'_best_checkpoint_seed' + str(
                                                            args.seed) +  '.model'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_data_path = '/home/user01/SP_OOD_Experiments/Waterbirds_dataset/waterbird_complete90_forest2water2'
    parser.add_argument("--data_path", type=str, default=default_data_path, help="data path")
    parser.add_argument("--dataset", type=str, default='Waterbirds')
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--backbone_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--sampling_mode", type=str, default='top-k')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--r", type=int, default=95)

    args = parser.parse_args()
    args = (args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    
    train_and_test_erm(args)