import numpy as np

def generate_distinct_colors(num_colors=5, min_distance=100):
    colors = []

    def l2_distance(color1, color2):
        return np.linalg.norm(np.array(color1) - np.array(color2))

    while len(colors) < num_colors:
        candidate = np.random.randint(0, 256, 3)  # Generate a random RGB color
        if all(l2_distance(candidate, color) >= min_distance for color in colors):
            colors.append(candidate)

    return np.array(colors)

colors = generate_distinct_colors()
print("Generated Colors:", colors)


class ColoredMNIST(Dataset):
    def __init__(self, train=True, transform=None, min_distance=100):
        self.mnist = datasets.MNIST(root='./data', train=train, download=True)
        self.transform = transform

        self.colors = generate_distinct_colors(num_colors=5, min_distance=min_distance)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]

        if label >= 5:
            return self.__getitem__((idx + 1) % len(self))  # Skip this item, use modulo to prevent out-of-bounds

        img_rgb = torch.cat([transforms.ToTensor()(img)] * 3, dim=0)

        background_color = self.colors[label]

        img_colored = background_color[:, None, None] * torch.ones_like(img_rgb)
        img_colored = torch.where(img_rgb > 0, img_rgb, img_colored)

        if self.transform:
            img_colored = self.transform(img_colored)

        return img_colored, label


train_dataset = ColoredMNIST(train=True, transform=transforms.ToTensor())
validation_dataset = ColoredMNIST(train=False, transform=transforms.ToTensor())