import os
import torch
from torchvision import transforms
from PIL import Image 
import tqdm
import numpy as np
import glob

root_path = 'dataset/'

emb_path = 'embeddings/'
os.makedirs(emb_path, exist_ok=True)

group_names = ['0_0', '0_1', '1_0', '1_1', 'm']
file_lists = {}
for name in group_names:
    f_list = glob.glob(root_path + name + '/*.jpg')
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


emb_dict = {}

for name in group_names:
    
    emb_list = []
    f_list = file_lists[name]
    for i in tqdm.tqdm(range(len(f_list))):
        
        image_path = f_list[i]
        with torch.no_grad():
            image = Image.open(image_path)
            image_tensor = transform(image)
            emb = model_dino(image_tensor.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            emb_list.append(emb)
    
    emb_dict[name] = np.array(emb_list, dtype='float32')

np.save(emb_path + 'embs.npy', emb_dict)
    

    