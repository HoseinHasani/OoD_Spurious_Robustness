import os
import pandas as pd
import random

def create_metadata(root_folder, output_csv='dataset_metadata.csv', val_ratio=0.2, test_ratio=0.2):
    metadata = []

    for class_name in os.listdir(root_folder):
        class_path = os.path.join(root_folder, class_name)
        if os.path.isdir(class_path):
            
            for attribute_name in os.listdir(class_path):
                attribute_path = os.path.join(class_path, attribute_name)
                if os.path.isdir(attribute_path):
                    
                    image_files = [f for f in os.listdir(attribute_path) 
                                   if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}]
                    
                    random.shuffle(image_files)
                    num_images = len(image_files)
                    val_count = int(num_images * val_ratio)
                    test_count = int(num_images * test_ratio)
                    
                    splits = [1] * val_count + [2] * test_count + [0] * (num_images - val_count - test_count)
                    
                    for i, image_file in enumerate(image_files):
                        image_path = os.path.join(attribute_path, image_file)
                        split = splits[i]
                        
                        metadata.append({
                            'address': image_path,
                            'class': class_name,
                            'attribute': attribute_name,
                            'split': split
                        })

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(output_csv, index=False)
    print(f"Metadata saved to {output_csv}")

root_folder = 'FSLSHIFT'  
create_metadata(root_folder)
