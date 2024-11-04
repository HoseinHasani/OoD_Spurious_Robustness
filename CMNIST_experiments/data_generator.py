import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np

def generate_distinct_colors(num_colors=5, min_distance=100):
    colors = []
    def l2_distance(color1, color2):
        return torch.norm(color1 - color2).item()
    
    while len(colors) < num_colors:
        candidate = torch.randint(0, 256, (3,)).float()  # Random RGB color
        if all(l2_distance(candidate, torch.tensor(color)) >= min_distance for color in colors):
            colors.append(candidate)
    
    return torch.stack(colors)  

class ColoredMNIST(Dataset):
    def __init__(self, colors, train=True, transform=None):
        self.mnist = datasets.MNIST(root='./data', train=train, download=True)
        self.transform = transform
        self.colors = colors

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        
        img_rgb = torch.cat([transforms.ToTensor()(img)] * 3, dim=0)
        
        if label < 5:
            background_color = self.colors[label]  
            in_distribution = True
        else:
            background_color = self.colors[label % 5] 
            in_distribution = False
        
        img_colored = background_color[:, None, None] * torch.ones_like(img_rgb)
        img_colored = torch.where(img_rgb > 0, img_rgb, img_colored)

        if self.transform:
            img_colored = self.transform(img_colored)
        
        return img_colored, label, in_distribution

colors = generate_distinct_colors(num_colors=5, min_distance=100)

train_dataset = ColoredMNIST(colors=colors, train=True, transform=transforms.ToTensor())
validation_dataset = ColoredMNIST(colors=colors, train=False, transform=transforms.ToTensor())
