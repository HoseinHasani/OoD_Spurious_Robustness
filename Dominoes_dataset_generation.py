import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sp_corr = 0.9

dataset_path = 'datasets'

train_cifar = CIFAR10(dataset_path, train=True, download=False)
test_cifar = CIFAR10(dataset_path, train=False, download=False)
train_mnist = MNIST(dataset_path, train=True, download=True)
test_mnist = MNIST(dataset_path, train=False, download=True)




