import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sp_corr = 0.9

n_maj = 1000
n_min = 500

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


cifar_cl_inds = [np.argwhere(np.array(train_cifar.targets) == cl).ravel() for cl in cifar_classes]
mnist_cl_inds = [np.argwhere(train_mnist.targets.numpy() == cl).ravel() for cl in cifar_classes]






