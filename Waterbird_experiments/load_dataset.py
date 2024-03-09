import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
import tqdm

dataset_path = 'waterbird/'
emb_path = 'waterbird_embs/'

os.makedirs(emb_path, exist_ok=True)


metadata = pd.read_csv(dataset_path + 'metadata.csv')

file_names = metadata['img_filename'].tolist()
labels = metadata['y'].tolist()
splits = metadata['split'].tolist()
places = metadata['place'].tolist()
place_filenames = metadata['place_filename'].tolist()



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

emb_dict = {}

for i in tqdm.tqdm(range(len(file_names))):
    name = f'{labels[i]}_{places[i]}_{splits[i]}'
    image_path = dataset_path + file_names[i]
    with torch.no_grad():
        image = Image.open(image_path)
        image_tensor = transform(image)
        emb = model_dino(image_tensor.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        emb_dict[name] = emb
        np.save(emb_path + name + '.npy', emb)
    
np.save('waterbird_embs.npy', emb_dict)


    
    
    
