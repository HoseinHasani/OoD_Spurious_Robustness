import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image 
import numpy as np
import os
import seaborn as sns



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




c11_prototype = c11_embs.mean(0)[None]
c12_prototype = c12_embs.mean(0)[None]
c21_prototype = c21_embs.mean(0)[None]
c22_prototype = c22_embs.mean(0)[None]
o1_prototype = o1_embs.mean(0)[None]
o2_prototype = o2_embs.mean(0)[None]

embs = np.concatenate([c11_embs, c12_embs, c21_embs, c22_embs, o1_embs, o2_embs])
protos = np.concatenate([c11_prototype, c12_prototype, c21_prototype, c22_prototype, o1_prototype, o2_prototype])

euc_dists = np.array([np.linalg.norm(embs - proto, axis=-1) for proto in protos])

euc_dists = np.round(euc_dists, 1).T

pic_path = 'pics/'
os.makedirs(pic_path, exist_ok=True)

plt.figure(figsize=(10, 10))
sns.heatmap(euc_dists, cmap='coolwarm', annot=True, linewidths=2)

plt.xlabel('prototypes', fontsize=16)
plt.ylabel('embeddings', fontsize=16)

plt.title('Unnormalized Euclidean Distance Matrix', fontsize=20)

plt.savefig(pic_path + 'euc_dist_from_prototypes.png', dpi=160)



embs_normalized = embs / np.linalg.norm(embs, axis=-1)[:, None]
protos_normalized = protos / np.linalg.norm(protos, axis=-1)[:, None]

cosine_dists = np.array([np.dot(embs_normalized, proto) for proto in protos_normalized])

cosine_dists = np.round(cosine_dists, 2).T

plt.figure(figsize=(10, 10))
sns.heatmap(cosine_dists, cmap='coolwarm', annot=True, linewidths=2)

plt.xlabel('prototypes', fontsize=16)
plt.ylabel('embeddings', fontsize=16)

plt.title('Cosine Distance Matrix', fontsize=20)

plt.savefig(pic_path + 'cosine_dist_mat.png', dpi=160)



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1)[:, None]

emb_probs = softmax(embs)
proto_probs = softmax(protos)

cross_entropy_losses = np.array([np.dot(np.log(emb_probs), proto) for proto in proto_probs])

cross_entropy_losses = - np.round(cross_entropy_losses, 2).T

plt.figure(figsize=(10, 10))
sns.heatmap(cross_entropy_losses, cmap='coolwarm', annot=True, linewidths=2)

plt.xlabel('prototypes', fontsize=16)
plt.ylabel('embeddings', fontsize=16)

plt.title('Cross-Entropy Matrix', fontsize=20)

plt.savefig(pic_path + 'Cross_entropy_mat.png', dpi=160)



concat_embs = np.concatenate([embs_normalized, protos_normalized])

tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=18)
tsne_embs = tsne.fit_transform(concat_embs)

concat_embs = np.concatenate([embs_normalized, protos_normalized])

tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10)
tsne_embs = tsne.fit_transform(concat_embs)

K = 6
#cmap = plt.get_cmap('jet')
#colors = np.random.permutation(K)
#colors = cmap(colors / np.max(colors) * 1)

colors = ['tab:blue', 'tab:green', 'tab:pink', 'tab:orange', 'tab:red', 'tab:cyan']

plt.figure(figsize=(7, 7))

for i in range(K):
    samples = tsne_embs[i * 4: (i + 1) * 4]
    proto = tsne_embs[-K + i]
    plt.scatter(samples[:, 0], samples[:, 1], marker='*', s=25, c=colors[i], label='samples ' + str(i))
    plt.scatter(proto[None, 0], proto[None, 1], marker='o', s=40, c=colors[i], label='prototype ' + str(i))    
  
plt.title('DINO t-SNE Embeddings')
plt.legend()
plt.savefig(pic_path + 'tsne.png', dpi=160)
