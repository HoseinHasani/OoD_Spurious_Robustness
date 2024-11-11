import os
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm


target_images = []

def image_difference(img1, img2):
    
    if img1.mode != img2.mode:
        # Convert both images to RGB only if their modes are different
        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')

    if img1.size != img2.size:
        return 200000
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
        for target_file, target_img in target_images:
            diff = image_difference(img, target_img)
            if diff < threshold:
                is_unique = False
                break

        if is_unique:
            n_copy += 1
            shutil.copy2(source_image_path, target_dir)
            target_images.append((source_file, img))  
        else:
            n_skip += 1

    print(f"\nFolder: {source_dir}, Copied: {n_copy}, Skipped: {n_skip}\n")


def deduplicate_dataset(root_source_dir, root_target_dir, threshold):

    for subdir in os.listdir(root_source_dir):
        source_subdir = os.path.join(root_source_dir, subdir)
        target_subdir = os.path.join(root_target_dir, subdir)

        if os.path.isdir(source_subdir):
            print(f"Processing directory: {source_subdir}")
            copy_unique_images(source_subdir, target_subdir, threshold)


root_source_directory = 'FSLSHIFT/lizard'  
root_target_directory = f'dedup/{root_source_directory}'  
deduplicate_dataset(root_source_directory, root_target_directory, threshold=56000)
