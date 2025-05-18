import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from tqdm import tqdm

batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_file = "cifar100_resnet50_embeddings.npy"

transform = transforms.Compose([
    transforms.Resize(224//4),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

cifar100_train = datasets.CIFAR100(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(cifar100_train, batch_size=batch_size, shuffle=False, num_workers=2)

resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity()  # Remove classification layer
resnet = resnet.to(device)
resnet.eval()

class_embeddings = defaultdict(list)
with torch.no_grad():
    for images, labels in tqdm(train_loader, desc="Extracting embeddings"):
        images = images.to(device)
        features = resnet(images)  # shape: [batch_size, 2048]
        for feature, label in zip(features.cpu().numpy(), labels.numpy()):
            class_embeddings[label].append(feature)

all_classes = sorted(class_embeddings.keys())
n_classes = len(all_classes)
feature_dim = len(class_embeddings[0][0])
samples_per_class = len(class_embeddings[0])

embeddings_array = np.zeros((n_classes, samples_per_class, feature_dim), dtype=np.float32)
for cls in all_classes:
    class_features = np.stack(class_embeddings[cls])
    embeddings_array[cls] = class_features

print(f"Embeddings shape: {embeddings_array.shape}")  # Should be (100, n_samples_per_class, 2048)

np.save(output_file, embeddings_array)
print(f"Saved embeddings to: {output_file}")
