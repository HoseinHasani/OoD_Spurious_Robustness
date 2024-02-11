import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sp_corr = 0.9

n_maj = 1000
n_min = 500
n_g = 1000

dataset_path = 'datasets'

train_cifar = CIFAR10(dataset_path, train=True, download=True)
test_cifar = CIFAR10(dataset_path, train=False, download=True)
train_mnist = MNIST(dataset_path, train=True, download=True)
test_mnist = MNIST(dataset_path, train=False, download=True)

mnist_classes = [0, 1]
cifar_classes = [0, 1, 8, 9]
core_classes = cifar_classes[:2]
ood_classes = cifar_classes[2:]

cifar_class_names = ['airplane', 'car', 'ship', 'truck']


cifar_cl_inds = [np.random.permutation(np.argwhere(np.array(train_cifar.targets) == cl).ravel()) for cl in cifar_classes]
mnist_cl_inds = [np.random.permutation(np.argwhere(train_mnist.targets.numpy() == cl).ravel()) for cl in mnist_classes]


for m in mnist_classes:
    for c, c_name in enumerate(cifar_classes):
        
        
        mnist_inds = mnist_cl_inds[m][c * n_g: (c + 1) * n_g]
        cifar_inds = cifar_cl_inds[c][:n_g]
        
        mnist_images0 = train_mnist.data[mnist_inds]
        cifar_images0 = train_cifar.data[cifar_inds]

        mnist_images = F.pad(mnist_images0, (2,2,2,2), value=0)[:, None].tile((1,3,1,1))
        cifar_images = torch.tensor(cifar_images0.transpose([0,3,1,2]))

        group_images = torch.cat([mnist_images, cifar_images], 3)
        group_name = f'{mnist_classes[m]}_{cifar_class_names[c]}'

        if True:
            plt.figure()
            plt.imshow(group_images[0].numpy().transpose([1,2,0]))
            plt.title(group_name)








