import os
import glob
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
import tqdm

dataset_path = 'OOD_Datasets/placesbg/'
emb_path = ''



place_names = ['water', 'land']

file_lists = {}
for name in place_names:
    f_list = glob.glob(dataset_path + name + '/*.jpg')
    print(name, len(f_list))
    file_lists[name] = f_list

model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model_dino.eval()

def center_pad(img, target_size):
    _, height, width = img.shape
    larger_shape = max(width, height)
    
    w_pad = (larger_shape - width) // 2
    h_pad = (larger_shape - height) // 2

    padded = transforms.functional.pad(img, (w_pad, h_pad, w_pad, h_pad),
                                     fill=0, padding_mode='constant')
    return padded


model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model_dino.eval()

device = torch.device("cuda")
model_dino.to(device)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda img: center_pad(img, 224)),
        transforms.Resize((224,224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])



for name in place_names:
    emb_dict = {}
    f_list = file_lists[name]
    for i in tqdm.tqdm(range(len(f_list))):
        key_name = f'{name}_{i}'
        image_path = f_list[i]
        with torch.no_grad():
            image = Image.open(image_path)
            image_tensor = transform(image)
            emb = model_dino(image_tensor.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            emb_dict[key_name] = emb
    np.save(name + '.npy', emb_dict)
    

    
    
    
