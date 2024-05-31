class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def load_embeddings(path):
    return np.load(path, allow_pickle=True).item()

def load_embeddings_and_labels(emb_path):
    emb_dict = np.load(emb_path, allow_pickle=True).item()
    train_emb, val_emb, test_emb = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    for name, emb in emb_dict.items():
        split = int(name.split('_')[2])
        label = int(name.split('_')[0])
        if split == 0:
            train_emb.append(emb)
            train_labels.append(label)
        elif split == 1:
            val_emb.append(emb)
            val_labels.append(label)
        elif split == 2:
            test_emb.append(emb)
            test_labels.append(label)

    return (np.array(train_emb), np.array(train_labels)), (np.array(val_emb), np.array(val_labels)), (np.array(test_emb), np.array(test_labels))

def load_embeddings_and_labels_clb(emb_path):
    emb_dict = np.load(emb_path, allow_pickle=True).item()
    train_emb, val_emb, test_emb = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    for name, emb in emb_dict.items():
        split = name.split('_')[2]
        label = int(name.split('_')[0])
        if split == 'train':
            train_emb.append(emb)
            train_labels.append(label)
        elif split == 'val':
            val_emb.append(emb)
            val_labels.append(label)
        elif split == 'test':
            test_emb.append(emb)
            test_labels.append(label)

    return (np.array(train_emb), np.array(train_labels)), (np.array(val_emb), np.array(val_labels)), (np.array(test_emb), np.array(test_labels))


def create_dataloaders(train_data, val_data, test_data, batch_size):
    train_loader = DataLoader(EmbeddingDataset(*train_data), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(EmbeddingDataset(*val_data), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(EmbeddingDataset(*test_data), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
