import os
import torch
from torchvision import transforms
from sklearn.manifold import TSNE
from PIL import Image 
import numpy as np
import utils

N_groups = 6
N_samples = 4 # number of samples per group

tsne_perplexity = int(N_groups * N_samples * 0.5)

group_names = ['c11', 'c12', 'c21', 'c22', 'o1', 'o2']
final_names = ['man_black', 'woman_black',
               'man_blond', 'woman_blond',
               'man_bald', 'woman_bald']

result_path = 'results/DINO_pics/'
image_path = 'face_toy_dataset/'

os.makedirs(result_path, exist_ok=True)

model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model_dino.eval()


device = torch.device("cuda")
model_dino.to(device)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])



image_list = []
embs_list = []
prototype_list = []

### Extract DINO embeddings: ####

grouped_embs = {}
for i, name in enumerate(group_names):
    
    imgs = [transform(Image.open(image_path + name + f' ({k + 1}).jpg')) for k in range(N_samples)]
    with torch.no_grad():
        embs = [model_dino(img.unsqueeze(0).to(device)).squeeze().cpu().numpy() for img in imgs]
    
    image_list.append(imgs)
    embs_list.append(np.array(embs))
    prototype_list.append(np.mean(embs, 0)[None])
    grouped_embs[final_names[i]] = np.array(embs)
    
np.save('face_embs', grouped_embs)

### Euclidean distance ####

cnct_embs = np.concatenate(embs_list)
cnct_protos = np.concatenate(prototype_list)

euc_dists = np.array([np.linalg.norm(cnct_embs - proto, axis=-1) for proto in cnct_protos])
euc_dists = np.round(euc_dists, 1).T



utils.plot_adj_mat(euc_dists, 'Unnormalized Euclidean Distance Matrix', result_path)


### Cosine distance ####

embs_normalized = cnct_embs / np.linalg.norm(cnct_embs, axis=-1)[:, None]
protos_normalized = cnct_protos / np.linalg.norm(cnct_protos, axis=-1)[:, None]

cosine_dists = np.array([np.dot(embs_normalized, proto) for proto in protos_normalized])
cosine_dists = np.round(cosine_dists, 2).T

utils.plot_adj_mat(cosine_dists, 'Cosine Similarity Matrix', result_path)


# cosine_dists = np.array([np.dot(embs_normalized, emb) for emb in embs_normalized])
# cosine_dists = np.round(cosine_dists, 2).T
# utils.plot_adj_mat(cosine_dists, 'Cosine (Sample) Distance Matrix', result_path)



### Cross-entropy distance ####

emb_probs = utils.softmax(cnct_embs)
proto_probs = utils.softmax(cnct_protos)

cross_entropy_losses = np.array([np.dot(np.log(emb_probs), proto) for proto in proto_probs])
cross_entropy_losses = - np.round(cross_entropy_losses, 2).T

utils.plot_adj_mat(cross_entropy_losses, 'Cross-Entropy Matrix', result_path)


### Apply t-SNE ####

all_embs = np.concatenate([embs_normalized, protos_normalized])

tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=tsne_perplexity)
tsne_embs = tsne.fit_transform(all_embs)

utils.plot_tsne(tsne_embs, final_names, N_groups, N_samples, 'DINO t-SNE Embeddings', result_path)


