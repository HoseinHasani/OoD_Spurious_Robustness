import os
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm


target_means = [] 

def image_mean(img):
    return np.mean(np.array(img), axis=(0, 1)) 

def copy_unique_images(source_dir, target_dir, threshold):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    source_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    

    n_copy = 0
    n_skip = 0

    for source_file in tqdm(source_files):
        source_image_path = os.path.join(source_dir, source_file)

        try:
            img = Image.open(source_image_path).convert('RGB')
            img_mean = image_mean(img)
        except Exception as e:
            print(f"Error loading image {source_file}: {e}")
            continue

        is_unique = True
        for target_mean in target_means:
            diff = np.max(np.abs(img_mean - target_mean))
            th_val = threshold * (1 - 0.5 * min(len(target_means) / 500, 1))
            if diff < th_val:
                is_unique = False
                break

        if is_unique:
            n_copy += 1
            shutil.copy2(source_image_path, target_dir)
            target_means.append(img_mean)  
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


root_source_directory = 'FSLSHIFT/sheep'  
root_target_directory = f'dedub/{root_source_directory}'  
deduplicate_dataset(root_source_directory, root_target_directory, threshold=1.6)
