import os
import shutil
from PIL import Image
import numpy as np

def load_image(image_path):
    """Load an image and return it as a numpy array."""
    try:
        img = Image.open(image_path)
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def calculate_image_difference(image1, image2):
    """Calculate pixel difference between two images."""
    if image1.shape != image2.shape:
        return np.inf  # Images of different sizes are considered different.
    # Compute pixel-wise absolute difference and take the sum of all differences.
    diff = np.sum(np.abs(image1.astype(np.int32) - image2.astype(np.int32)))
    return diff

def move_unique_images(source_dir, dest_dir, threshold=1000):
    """Move unique images from source_dir to dest_dir based on raw pixel comparison."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    image_files = []
    # Gather image files with common image extensions
    for ext in ['jpeg', 'jpg', 'png', 'bmp', 'tiff']:
        image_files.extend([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(ext)])

    unique_images = []

    for i, img_path in enumerate(image_files):
        img1 = load_image(img_path)
        if img1 is None:
            continue

        is_duplicate = False
        for unique_img_path in unique_images:
            img2 = load_image(unique_img_path)
            if img2 is None:
                continue
            diff = calculate_image_difference(img1, img2)
            if diff < threshold:  # If difference is below the threshold, consider as duplicate
                print(f"Duplicate found: {img_path} is similar to {unique_img_path}, difference: {diff}")
                is_duplicate = True
                break

        if not is_duplicate:
            unique_images.append(img_path)
            # Move the unique image to the destination directory
            shutil.move(img_path, os.path.join(dest_dir, os.path.basename(img_path)))
            print(f"Moved unique image: {img_path}")

# Example usage:
source_directory = '/path/to/source_directory'  # Replace with your source directory
destination_directory = '/path/to/destination_directory'  # Replace with your destination directory
move_unique_images(source_directory, destination_directory, threshold=1000)
