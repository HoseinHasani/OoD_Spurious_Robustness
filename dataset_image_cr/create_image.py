import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

data_path = 'data'
class_folders = ['Class 1', 'Class 2', 'Class 3', 'Near-OOD']
contexts = ['1', '2', '3']

num_classes = len(class_folders) - 1  
num_contexts = len(contexts)
fig, axes = plt.subplots(num_contexts, len(class_folders), figsize=(12, 8))

for col, class_folder in enumerate(class_folders):
    for row, context in enumerate(contexts):
        image_path = os.path.join(data_path, class_folder, f"{context}.jpg")
        
        if os.path.exists(image_path):
            img = Image.open(image_path)
            axes[row, col].imshow(img)
        else:
            axes[row, col].text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', color='red')
        
        axes[row, col].axis('off')

        if row == 0:
            axes[row, col].set_title(class_folder, fontsize=14, weight='bold')

highlight_positions = [(0, 0), (1, 1), (2, 2)]  
for (row, col) in highlight_positions:
    rect = patches.Rectangle((0, 0), 1, 1, transform=axes[row, col].transAxes,
                             edgecolor='blue', linestyle='dotted', linewidth=9, fill=False)
    axes[row, col].add_patch(rect)

sep_position = num_classes / len(class_folders) * 0.99
fig.add_artist(plt.Line2D([sep_position, sep_position], [0.01, 0.99], transform=fig.transFigure, color='black', linestyle='--', linewidth=4))

plt.tight_layout()
plt.savefig('dataset_example.pdf')
plt.show()
