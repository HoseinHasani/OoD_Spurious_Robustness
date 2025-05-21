import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_file = "svhn_resnet50_ood_embeddings.npy"

transform = transforms.Compose([
    transforms.Resize(224 // 4),  # Resize to 56x56
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

svhn_data = datasets.SVHN(root="./data", split="train", transform=transform, download=True)
svhn_loader = DataLoader(svhn_data, batch_size=batch_size, shuffle=False, num_workers=2)

resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity()
resnet = resnet.to(device)
resnet.eval()

all_embeddings = []

with torch.no_grad():
    for images, _ in tqdm(svhn_loader, desc="Extracting SVHN embeddings"):
        images = images.to(device)
        features = resnet(images)
        all_embeddings.append(features.cpu().numpy())

all_embeddings = np.vstack(all_embeddings)
np.random.shuffle(all_embeddings)

subset_embeddings = all_embeddings[:5000]
np.save(output_file, subset_embeddings)
print(f"Saved {subset_embeddings.shape[0]} SVHN embeddings to: {output_file}")
