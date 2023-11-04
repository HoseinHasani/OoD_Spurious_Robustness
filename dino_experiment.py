import torch
from torchvision import transforms
from PIL import Image 
import numpy as np


image_path = 'face_toy_dataset/'

model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model_dino.eval()


device = torch.device("cuda")
model_dino.to(device)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

c11_imgs = [transform(Image.open(image_path + 'c11 (' + str(k + 1) + ').jpg')) for k in range(4)]
c12_imgs = [transform(Image.open(image_path + 'c12 (' + str(k + 1) + ').jpg')) for k in range(4)]
c21_imgs = [transform(Image.open(image_path + 'c21 (' + str(k + 1) + ').jpg')) for k in range(4)]
c22_imgs = [transform(Image.open(image_path + 'c22 (' + str(k + 1) + ').jpg')) for k in range(4)]
o1_imgs = [transform(Image.open(image_path + 'o1 (' + str(k + 1) + ').jpg')) for k in range(4)]
o2_imgs = [transform(Image.open(image_path + 'o2 (' + str(k + 1) + ').jpg')) for k in range(4)]


with torch.no_grad():
    c11_embs = np.array([model_dino(img.unsqueeze(0).to(device)).squeeze().cpu().numpy() for img in c11_imgs])
    c12_embs = np.array([model_dino(img.unsqueeze(0).to(device)).squeeze().cpu().numpy() for img in c12_imgs])
    c21_embs = np.array([model_dino(img.unsqueeze(0).to(device)).squeeze().cpu().numpy() for img in c21_imgs])
    c22_embs = np.array([model_dino(img.unsqueeze(0).to(device)).squeeze().cpu().numpy() for img in c22_imgs])
    o1_embs = np.array([model_dino(img.unsqueeze(0).to(device)).squeeze().cpu().numpy() for img in o1_imgs])
    o2_embs = np.array([model_dino(img.unsqueeze(0).to(device)).squeeze().cpu().numpy() for img in o2_imgs])

