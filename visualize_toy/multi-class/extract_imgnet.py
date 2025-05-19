import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

root_dir = 'R/Datasets/OOD_Datasets/spurious_imagenet/dataset/spurious_imagenet/ID_classes/'  
samples_per_folder = 40
output_file = 'SPI_ID_resnet50.npy'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

model = resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  
model.eval().to(device)

folder_names = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
embeddings = []

with torch.no_grad():
    for folder in tqdm(folder_names, desc="Processing folders"):
        folder_path = os.path.join(root_dir, folder)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        selected_images = random.sample(image_files, min(samples_per_folder, len(image_files)))
        
        folder_embeddings = []

        for img_file in selected_images:
            img_path = os.path.join(folder_path, img_file)
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            embedding = model(image_tensor).squeeze().cpu().numpy()  
            folder_embeddings.append(embedding)

        while len(folder_embeddings) < samples_per_folder:
            print('eeeeeeeee')
            folder_embeddings.append(np.zeros_like(folder_embeddings[0]))

        embeddings.append(folder_embeddings)

embeddings_np = np.array(embeddings)  
np.save(output_file, embeddings_np)
print(f"Saved embeddings to {output_file}")
