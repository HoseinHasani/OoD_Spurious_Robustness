import os
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm


def image_difference(img1, img2):

    if img1.size != img2.size:
        return 100000  
    diff = np.sum(np.abs(np.array(img1) - np.array(img2)))
    
    return diff


def copy_unique_images(source_dir, target_dir, threshold):

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    source_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    n_copy = 0
    n_skip = 0
    
    for i in tqdm(range(len(source_files))):
        source_file = source_files[i]
        source_image_path = os.path.join(source_dir, source_file)
        
        try:
            img = Image.open(source_image_path)
        except Exception as e:
            print(f"Error loading image {source_file}: {e}")
            continue
        
        is_unique = True
        for target_file in os.listdir(target_dir):
            target_image_path = os.path.join(target_dir, target_file)
            target_img = Image.open(target_image_path)

            diff = image_difference(img, target_img)
            if diff < threshold:
                is_unique = False
                break
        
        if is_unique:
            n_copy += 1
            shutil.copy2(source_image_path, target_dir)
        else:
            n_skip += 1
            
    print()
    print(source_dir, n_skip, n_copy)
    
    
source_directory = 'black'
target_directory = 'black2'
copy_unique_images(source_directory, target_directory, threshold=6000)
