import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd


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
    width, height = img.size
    left_pad = max(0, (target_size - width) // 2)
    right_pad = max(0, target_size - width - left_pad)
    top_pad = max(0, (target_size - height) // 2)
    bottom_pad = max(0, target_size - height - top_pad)

    return transforms.functional.pad(img, (left_pad, top_pad, right_pad, bottom_pad),
                                     fill=111, padding_mode='constant')


model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model_dino.eval()

device = torch.device("cuda")
model_dino.to(device)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Lambda(lambda img: center_pad(img, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

emb_dict = {}

for i in range(len(file_names)):
    name = f'{labels[i]}_{places[i]}_{splits[i]}'
    image_path = dataset_path + file_names[i]
    image = Image.open(image_path)
    transformed_image = transform(image)
    emb = model_dino(transformed_image.unsqueeze(0).to(device)).squeeze().cpu().numpy()
    emb_dict[name] = emb
    np.save(emb_path + name + '.npy', emb)
    
np.save('waterbird_embs.npy', emb_dict)


    
    
    
