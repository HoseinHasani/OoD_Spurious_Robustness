import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


data_path = 'data'
class_folders = ['Class 1', 'Class 2', 'Class 3', 'SP-OOD', 'NSP-OOD']
contexts = ['1', '2', '3']

num_classes = len(class_folders) - 2
num_contexts = len(contexts)
fig, axes = plt.subplots(num_contexts, len(class_folders), figsize=(14, 8))

for col, class_folder in enumerate(class_folders):
    for row, context in enumerate(contexts):
        image_path = os.path.join(data_path, class_folder, f"{context}.jpg")
        
        if os.path.exists(image_path):
            img = Image.open(image_path)
            width, height = img.size 
  
            new_width = int(width * 1.065)
            new_height = int(height * 1.065)
            left = (new_width - width) // 2
            top = (new_height - height) // 2
              
            result = Image.new(img.mode, (new_width, new_height), (255, 255, 255)) 
            result.paste(img, (left, top))
            
            axes[row, col].imshow(result)
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

sep_position = num_classes / len(class_folders) * 0.996
fig.add_artist(plt.Line2D([sep_position, sep_position], [0.025, 0.96], transform=fig.transFigure,
                          color='black', linestyle='--', linewidth=3.4))

plt.tight_layout()
plt.savefig('dataset_example.pdf')
plt.show()
