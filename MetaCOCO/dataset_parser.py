import os
import pandas as pd

def get_image_count_in_folder(folder_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    return sum(1 for file in os.listdir(folder_path) if os.path.splitext(file)[1].lower() in image_extensions)

def parse_dataset_folder(root_folder):
    classes = {}
    attributes = set()
    
    for class_name in os.listdir(root_folder):
        class_path = os.path.join(root_folder, class_name)
        if os.path.isdir(class_path):
            class_data = {}
            
            for attribute_name in os.listdir(class_path):
                attribute_path = os.path.join(class_path, attribute_name)
                if os.path.isdir(attribute_path):
                    image_count = get_image_count_in_folder(attribute_path)
                    class_data[attribute_name] = image_count
                    attributes.add(attribute_name)
            
            classes[class_name] = class_data
    
    attributes = sorted(attributes)
    df = pd.DataFrame(index=attributes, columns=sorted(classes.keys())).fillna(0)

    for class_name, class_data in classes.items():
        for attribute_name, image_count in class_data.items():
            df.loc[attribute_name, class_name] = image_count

    df = df.astype(int)

    df['Total'] = df.sum(axis=1)
    df.loc['Total'] = df.sum(axis=0)

    ratio_df = df.copy()

    for class_name in df.columns[:-1]:  
        ratio_df[class_name] = (df[class_name] / df[class_name].sum() * 200).round().astype(int)

    ratio_df['Total'] = (df['Total'] / df['Total'].sum() * 100).round().astype(int)

    return df, ratio_df

def save_stats_to_csv(root_folder, output_csv='dataset_statistics.csv', ratio_csv='dataset_ratio_statistics.csv'):
    df, ratio_df = parse_dataset_folder(root_folder)
    
    df.to_csv(output_csv)
    print(f"Statistics saved to {output_csv}")
    
    ratio_df.to_csv(ratio_csv)
    print(f"Ratio statistics saved to {ratio_csv}")

root_folder = 'FSLSHIFT'  
save_stats_to_csv(root_folder)
