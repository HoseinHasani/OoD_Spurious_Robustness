import csv
import pandas as pd


dataset_path = 'waterbird/'

metadata = pd.read_csv(dataset_path + 'metadata.csv')

file_names = metadata['img_filename'].tolist()
labels = metadata['y'].tolist()
splits = metadata['split'].tolist()
places = metadata['place'].tolist()
place_filenames = metadata['place_filename'].tolist()

