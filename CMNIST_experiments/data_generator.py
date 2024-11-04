import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np

def generate_distinct_colors(num_colors=5, min_distance=100):
    colors = []
    def l2_distance(color1, color2):
        return torch.norm(color1 - color2).item()
    
    while len(colors) < num_colors:
        candidate = torch.randint(0, 256, (3,)).float()  
        if all(l2_distance(candidate, torch.tensor(color)) >= min_distance for color in colors):
            colors.append(candidate)
    
    return torch.stack(colors)  
class ColoredMNIST(Dataset):
    def __init__(self, colors, train=True, transform=None, sp_ratio=0.90):
        self.mnist = datasets.MNIST(root='./data', train=train, download=True)
        self.transform = transform
        self.colors = colors
        self.train = train
        self.sp_ratio = sp_ratio

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        
        img_rgb = torch.cat([transforms.ToTensor()(img)] * 3, dim=0)
        
        if label < 5:
            if self.train:
                if torch.rand(1).item() < self.sp_ratio:
                    # Majority group: default color for the class
                    background_color = self.colors[label]
                else:
                    # Minority group: use colors from two other classes
                    alt_colors = [c for c in range(5) if c != label]
                    chosen_alt_color = np.random.choice(alt_colors)
                    background_color = self.colors[chosen_alt_color]
            else:
                # For validation, each color appears with equal probability (0.2 for each color)
                color_index = np.random.choice(range(5), p=[0.2]*5)
                background_color = self.colors[color_index]

            in_distribution = True
        else:
            # Out-of-distribution samples (digits 5 to 9)
            background_color = self.colors[label % 5]
            in_distribution = False
        
        img_colored = background_color[:, None, None] * torch.ones_like(img_rgb)
        img_colored = torch.where(img_rgb > 0, img_rgb, img_colored)

        if self.transform:
            img_colored = self.transform(img_colored)
        
        return img_colored, label, in_distribution

colors = generate_distinct_colors(num_colors=5, min_distance=100)

train_dataset = ColoredMNIST(colors=colors, train=True, transform=transforms.ToTensor(), sp_ratio=0.90)
validation_dataset = ColoredMNIST(colors=colors, train=False, transform=transforms.ToTensor())
