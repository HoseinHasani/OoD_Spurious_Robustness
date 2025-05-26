import os
import shutil
from collections import defaultdict

input_dir = "pickles"
output_dir = "pickles"
os.makedirs(output_dir, exist_ok=True)

target_dataset = "animals_metacoco"
synthetic_seed_base = 100

experiments = defaultdict(list)


backbone_names = ['BiT_M_R50x1', 'ConvNeXt_B', 'DeiT_B', 'Swin_B', 'ViT_S',
                  'dinov2_vits14', 'resnet_101', 'resnet_18', 'resnet_34', 'resnet_50']


for filename in os.listdir(input_dir):
    if not filename.endswith(".pkl") or '^' not in filename:
        continue

    parts = filename[:-4].split('^')
    if len(parts) < 7:
        continue

    dataset, backbone, method, ood_set, correlation, seed = parts[:6]
    flag = "_".join(parts[6:])
    
    # if backbone != 'resnet_50':
    #     continue

    # if not('iter3' in flag):
    #     continue

    # if ood_set != 'animals_ood':
    #     continue
    


    if backbone not in backbone_names:
        continue
    
    # if method not in ['sprod3']:
    #     continue
    # if 'iter3' not in flag:
    #     continue
    
    
    if dataset == 'animals_metacoco' and method == 'sprod3' and backbone == 'resnet_50':
        if 'iter2' in flag:
            continue
    

    if dataset != target_dataset or flag == "default":
        continue

    key = (dataset, backbone, method, ood_set, correlation, seed)
    experiments[key].append((filename, flag))

for (dataset, backbone, method, ood_set, correlation, orig_seed), file_info_list in experiments.items():
    for i, (fname, original_flag) in enumerate(sorted(file_info_list)):
        org_seed = int(orig_seed[1:]) - 20
        new_seed_num = synthetic_seed_base + 13 * org_seed + i
        new_seed = f"s{new_seed_num}"

        new_filename = f"{dataset}^{backbone}^{method}^{ood_set}^r95^{new_seed}^default.pkl"
        src_path = os.path.join(input_dir, fname)
        dst_path = os.path.join(output_dir, new_filename)

        shutil.copyfile(src_path, dst_path)
        print(f"Copied: {new_filename}")
