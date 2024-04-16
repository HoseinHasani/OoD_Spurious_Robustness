import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import os
import tqdm
from matplotlib import pyplot as plt
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_maj = 4000
n_min = 1000

dataset_path = 'datasets'

train_cifar = CIFAR10(dataset_path, train=True, download=True)
test_cifar = CIFAR10(dataset_path, train=False, download=True)
train_mnist = MNIST(dataset_path, train=True, download=True)
test_mnist = MNIST(dataset_path, train=False, download=True)

mnist_classes = [0, 1]
cifar_classes = [1, 9]

cifar_class_names = ['automobile', 'truck']


cifar_cl_inds = [np.random.permutation(np.argwhere(np.array(train_cifar.targets) == cl).ravel()) for cl in cifar_classes]
mnist_cl_inds = [np.random.permutation(np.argwhere(train_mnist.targets.numpy() == cl).ravel()) for cl in mnist_classes]

cifar_ood_inds = [np.random.permutation(np.argwhere(np.array(test_cifar.targets) == cl).ravel()) for cl in cifar_classes]

  
grouped_imgs = {}
ood_imgs = {}

def make_batch_images(mnist_class, cifar_class, mnist_inds, cifar_inds):
    
    
    mnist_inds = mnist_cl_inds[mnist_class][mnist_inds]
    cifar_inds = cifar_cl_inds[cifar_class][cifar_inds]
    
    mnist_images0 = train_mnist.data[mnist_inds]
    cifar_images0 = train_cifar.data[cifar_inds]
    
    mnist_images = F.pad(mnist_images0, (2, 2, 2, 2), value=0)[:, None].tile((1,3,1,1))
    cifar_images = torch.tensor(cifar_images0.transpose([0, 3, 1, 2]))
    
    group_images0 = torch.cat([mnist_images, cifar_images], 2)
    group_name = f'{mnist_classes[mnist_class]}_{cifar_class_names[cifar_class]}'
    
    group_images = F.pad(group_images0, (48, 48, 32, 32), value=120)
    grouped_imgs[group_name] = group_images

    if True:
        plt.figure()
        plt.imshow(group_images[0].numpy().transpose([1, 2, 0]))
        plt.title(group_name)
        

def make_batch_images_ood(cifar_class):
    
    
    cifar_images0 = test_cifar.data[cifar_ood_inds[cifar_class]]
    
    cifar_images = torch.tensor(cifar_images0.transpose([0, 3, 1, 2]))
    mnist_images = torch.zeros_like(cifar_images)
    group_images0 = torch.cat([mnist_images, cifar_images], 2)
    group_name = f'OOD_{cifar_class_names[cifar_class]}'
    
    group_images = F.pad(group_images0, (48, 48, 32, 32), value=120)
    ood_imgs[group_name] = group_images

    if True:
        plt.figure()
        plt.imshow(group_images[0].numpy().transpose([1, 2, 0]))
        plt.title(group_name)
        


make_batch_images(0, 0, np.arange(0, n_maj), np.arange(0, n_maj))
make_batch_images(1, 0, np.arange(n_maj, n_maj + n_min), np.arange(n_maj, n_maj + n_min))

make_batch_images(1, 1, np.arange(0, n_maj), np.arange(0, n_maj))
make_batch_images(0, 1, np.arange(n_maj, n_maj + n_min), np.arange(n_maj, n_maj + n_min))


make_batch_images_ood(0)
make_batch_images_ood(1)

model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model_dino.eval()

device = torch.device("cuda")
model_dino.to(device)

transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    
grouped_embs = {}

for g_name in grouped_imgs.keys():
    
    g_data = grouped_imgs[g_name]
    embs_list = []
    for data in tqdm.tqdm(g_data):
        with torch.no_grad():
            in_data = transform(data.to(device)[None]/255.)
            embs = model_dino(in_data).squeeze().cpu().numpy()
            embs_list.append(embs)
    embs_list = np.array(embs_list)
    grouped_embs[g_name] = embs_list


#with open('Dominoes_grouped_embs.pkl', 'wb') as f:
#    pickle.dump(grouped_embs, f)
    
np.save('Dominoes_train_embs.npy', grouped_embs)


ood_embs = {}

for g_name in ood_imgs.keys():
    
    g_data = ood_imgs[g_name]
    embs_list = []
    for data in tqdm.tqdm(g_data):
        with torch.no_grad():
            in_data = transform(data.to(device)[None]/255.)
            embs = model_dino(in_data).squeeze().cpu().numpy()
            embs_list.append(embs)
    embs_list = np.array(embs_list)
    ood_embs[g_name] = embs_list

np.save('Dominoes_ood_embs.npy', ood_embs)
