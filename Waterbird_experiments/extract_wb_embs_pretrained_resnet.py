import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
import tqdm

resnet_type = 50
dataset_path = 'waterbird/'
emb_path = f'wb_embs_res{resnet_type}/'
os.makedirs(emb_path, exist_ok=True)


metadata = pd.read_csv(dataset_path + 'metadata.csv')

img_ids = metadata['img_id'].tolist()
file_names = metadata['img_filename'].tolist()
labels = metadata['y'].tolist()
splits = metadata['split'].tolist()
places = metadata['place'].tolist()
place_filenames = metadata['place_filename'].tolist()


def center_pad(img, target_size):
    _, height, width = img.shape
    larger_shape = max(width, height)
    
    w_pad = (larger_shape - width) // 2
    h_pad = (larger_shape - height) // 2

    padded = transforms.functional.pad(img, (w_pad, h_pad, w_pad, h_pad),
                                     fill=0, padding_mode='constant')
    return padded

device = torch.device("cuda")
model0 = torch.hub.load('pytorch/vision:v0.10.0', f'resnet{resnet_type}', pretrained=True)
model = torch.nn.Sequential(*list(model0.children())[:-1])
model.to(device)
model.eval()



transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda img: center_pad(img, 224)),
        transforms.Resize((224,224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

emb_dict = {}

for i in tqdm.tqdm(range(len(file_names))):
    name = f'{labels[i]}_{places[i]}_{splits[i]}_{img_ids[i]}'
    image_path = dataset_path + file_names[i]
    with torch.no_grad():
        image = Image.open(image_path)
        image_tensor = transform(image)
        emb = model(image_tensor.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        emb_dict[name] = emb
        # np.save(emb_path + name + '.npy', emb)
    
np.save(f'wb_embs_res{resnet_type}_pretrained.npy', emb_dict)
