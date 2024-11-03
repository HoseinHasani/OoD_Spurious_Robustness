import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

split = "train"
sp_corr = 0.5 # 0.5 or 0.95

sp_type = "bg_co_occur" # bg_co_occur or bg_only

dataset_dir = f"data/urbancars/bg-{sp_corr}_co_occur_obj-{sp_corr}"
resnet_type = 50  

if resnet_type == 18:
    model = models.resnet18(pretrained=True)
elif resnet_type == 50:
    model = models.resnet50(pretrained=True)
else:
    raise ValueError("Only ResNet-18 and ResNet-50 are supported.")

model = nn.Sequential(*list(model.children())[:-1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_path

def extract_embeddings_batch(dataloader, model):
    embeddings = {}
    model.eval()
    with torch.no_grad():
        for images, image_paths in tqdm(dataloader, desc="Extracting embeddings", unit="batch"):
            images = images.to(device)
            outputs = model(images).squeeze(-1).squeeze(-1) 
            outputs = outputs.cpu().numpy()  
            for output, image_path in zip(outputs, image_paths):
                folder_name = os.path.basename(os.path.dirname(image_path))
                if folder_name not in embeddings:
                    embeddings[folder_name] = []
                embeddings[folder_name].append(output)
    return embeddings


image_paths_dict = {}
for group_name in os.listdir(dataset_dir):
    subfolder_path = os.path.join(dataset_dir, group_name)
    if os.path.isdir(subfolder_path):
        image_paths_dict[group_name] = [os.path.join(subfolder_path, image_name) for image_name in os.listdir(subfolder_path) if image_name.endswith(f"{sp_type}.png")]

all_image_paths = []
for subfolder, image_paths in image_paths_dict.items():
    all_image_paths.extend(image_paths)

batch_size = 64
dataset = ImageDataset(all_image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

embeddings_dict = extract_embeddings_batch(dataloader, model)

for folder_name in embeddings_dict:
    embeddings_dict[folder_name] = np.array(embeddings_dict[folder_name])

for key in embeddings_dict.keys():
    print(key, print(len(embeddings_dict[key])))

np.save(f"urbancars_{split}_{sp_type}_res{resnet_type}_pretrained.npy", embeddings_dict)

