import os
import numpy as np
import random
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm
from dataset_utils import crop_and_resize, combine_and_mask

places_dir = 'raw_datasets/places365_standard'

target_places = ['bamboo_forest', 'forest/broadleaf', 'ocean', 'lake/natural']

place_ids_df = pd.read_csv(
    os.path.join(places_dir, 'categories_places365.txt'),
    sep=" ",
    header=None,
    names=['place_name', 'place_id'],
    index_col='place_id')

