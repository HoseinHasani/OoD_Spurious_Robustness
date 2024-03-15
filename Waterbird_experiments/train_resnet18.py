import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import resnet

n_epochs = 40
batch_size = 64
n_ensemble = 16

st_rej_prob = 0.5
end_rej_prob = 0.05

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)


dataset_path = 'dataset'
log_path_root = 'res18_log_arrays'


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_cifar = CIFAR100(dataset_path, train=True, download=True, transform=transform_train)
test_cifar = CIFAR100(dataset_path, train=False, download=True, transform=transform_test)


trainloader = torch.utils.data.DataLoader(train_cifar, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_cifar, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    

net = resnet.ResNet18().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

for epoch in range(n_epochs):
    
    rej_prob = st_rej_prob - epoch / n_epochs * (st_rej_prob - end_rej_prob)
    
    net.train()
    for i, data in enumerate(trainloader, 0):
        
        if np.random.rand() < rej_prob:
            continue
        
        inputs, labels = data
        
        
        optimizer.zero_grad()

        outputs = net(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        
    scheduler.step()
    
    accs = []
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            accs.append((predicted == labels.to(device)).to(torch.float32).mean().item())
    
    print(f'Epoch: {epoch}, Acc: {100 * np.round(np.mean(accs), 4)} %')
                
    
all_outputs = []
all_labels = []

net.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        
        all_outputs.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
                
all_outputs = np.concatenate(all_outputs, 0)
all_labels = np.concatenate(all_labels, 0).astype('uint8')

log_path = log_path_root
os.makedirs(log_path, exist_ok=True)

np.save(log_path + 'preds.npy', all_outputs)
np.save(log_path + 'labels.npy', all_labels)
    
    
    

