import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def generate_distinct_colors(num_colors=5, min_distance=120, automatic=False):
    colors = []
    def l2_distance(color1, color2):
        return torch.norm(color1 - color2).item()
    
    if automatic:
        while len(colors) < num_colors:
            candidate = torch.randint(0, 180, (3,)).float()  
            if all(l2_distance(candidate, torch.tensor(color)) >= min_distance for color in colors):
                colors.append(candidate)
    else:
        colors = [
            torch.tensor([10, 190, 10]),
            torch.tensor([190, 20, 20]),
            torch.tensor([130, 10, 150]),
            torch.tensor([10, 20, 190]),
            torch.tensor([200, 150, 0]),
            ]
    return torch.stack(colors)  


class ColoredMNIST(Dataset):
    def __init__(self, colors, train=True, transform=None, sp_ratio=0.90):
        self.mnist = datasets.MNIST(root='./data', train=train, download=True)
        self.transform = transform
        self.colors = colors
        self.train = train
        self.sp_ratio = sp_ratio
        self.to_pil = ToPILImage()

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        
        img_rgb = torch.cat([transforms.ToTensor()(img)] * 3, dim=0)

        if label < 5:
            if self.train:
                if torch.rand(1).item() < self.sp_ratio:
                    # Majority group: default color for the class
                    color_index = label
                else:
                    # Minority group: use colors from two other classes
                    alt_colors = [c for c in range(5) if c != label]
                    color_index = np.random.choice(alt_colors)
            else:
                # For validation, each color appears with equal probability (0.2 for each color)
                color_index = np.random.choice(range(5), p=[0.2] * 5)
    
            in_distribution = True
        else:
            # Out-of-distribution samples (digits 5 to 9)
            color_index = label % 5
            in_distribution = False
        
        background_color = self.colors[color_index] / 256
        img_colored = background_color[:, None, None] * torch.ones_like(img_rgb)
        img_colored = torch.where(img_rgb > 0.24, img_rgb, img_colored)


        if self.transform:
            img_colored = self.to_pil(img_colored)
            img_colored = self.transform(img_colored)
            
        return img_colored, label, in_distribution, color_index


colors = generate_distinct_colors(num_colors=5)

train_dataset = ColoredMNIST(colors=colors, train=True, transform=None, sp_ratio=0.90)
validation_dataset = ColoredMNIST(colors=colors, train=False, transform=None)



def visualize_colored_mnist(dataset, ood_dataset, num_samples=1):
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    fig.suptitle("In-Distribution Samples", fontsize=16)
    
    to_pil = ToPILImage()
    
    for label in range(5):  
        color_indices = [0, 1, 2, 3, 4] 
        for i, color_idx in enumerate(color_indices):
            found = 0
            for img, lbl, in_dist, sample_color_idx in dataset:
                if np.random.rand() < 0.1:
                    continue
                if lbl == label and in_dist and sample_color_idx == color_idx:
                    axes[label, i].imshow(to_pil(img))
                    axes[label, i].axis("off")
                    found += 1
                    if found >= num_samples:
                        break
            else:
                axes[label, i].axis("off")
    
    for ax, col in zip(axes[0], color_indices):
        ax.set_title(f"Color {col}", fontsize=10)

    for ax, row in zip(axes[:, 0], range(5)):
        ax.set_ylabel(f"Class {row}", rotation=0, labelpad=40, fontsize=12, ha='right', va='center')

    for idx in range(5):
        axes[idx, idx].add_patch(
            plt.Rectangle((0, 0), 1, 1, transform=axes[idx, idx].transAxes,
                          edgecolor="red", linewidth=10, fill=False, linestyle='dotted')
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.show()
    
    ood_fig, ood_axes = plt.subplots(1, 5, figsize=(12, 3))
    ood_fig.suptitle("Out-of-Distribution Samples", fontsize=16)
    
    for i, label in enumerate(range(5, 10)):
        for img, lbl, in_dist, sample_color_idx in ood_dataset:
            if np.random.rand() < 0.1:
                continue
            if lbl == label and not in_dist:
                ood_axes[i].imshow(to_pil(img))
                ood_axes[i].set_title(f"OOD Class {label}", fontsize=10)
                ood_axes[i].axis("off")
                break

    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.show()

# visualize_colored_mnist(train_dataset, validation_dataset)

if False:
    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    train_dataset.transform = resize_transform
    validation_dataset.transform = resize_transform

model = resnet18(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if False:
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model.fc = nn.Linear(model.fc.in_features, 5) 
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            mask = labels < 5
            images, labels = images[mask], labels[mask]
            
            if labels.size(0) == 0:
                continue
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / total
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")




model.fc = nn.Identity()  
model.eval()

model.to(device)

def extract_embeddings(dataset, model, device):
    embeddings_dict = {}
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    with torch.no_grad():
        for img_colored, label, _, color_index in tqdm(dataloader):
            img_colored = img_colored.to(device)
            # padded = nn.functional.pad(img_colored, (98, 98, 98, 98), "constant", 0)
            embeddings = model(img_colored).cpu().numpy()
            for i in range(embeddings.shape[0]):
                key = f"{label[i].item()}_{color_index[i].item()}"
                if key not in embeddings_dict:
                    embeddings_dict[key] = []
                embeddings_dict[key].append(embeddings[i])
    
    for key in embeddings_dict:
        embeddings_dict[key] = np.array(embeddings_dict[key])
    
    return embeddings_dict

emb_path = 'embeddings'
os.makedirs(emb_path, exist_ok=True)
train_embeddings = extract_embeddings(train_dataset, model, device)
np.save(f"{emb_path}/cmnist_train_res18_pretrained.npy", train_embeddings)

val_embeddings = extract_embeddings(validation_dataset, model, device)
np.save(f"{emb_path}/cmnist_val_res18_pretrained.npy", val_embeddings)

