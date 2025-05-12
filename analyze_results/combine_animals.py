import os
import pickle
import numpy as np
from collections import defaultdict

input_dir = "pickles"
output_dir = "pickles_averaged"
os.makedirs(output_dir, exist_ok=True)

target_dataset = "animals_metacoco"

experiments = defaultdict(list)

for filename in os.listdir(input_dir):
    if not filename.endswith(".pkl") or '^' not in filename:
        continue

    parts = filename[:-4].split('^')
    if len(parts) < 7:
        continue

    dataset, backbone, method, ood_set, correlation, seed = parts[:6]
    flag_parts = parts[6:]

    if dataset != target_dataset:
        continue

    key = (dataset, backbone, method, ood_set, correlation, seed)
    experiments[key].append(filename)

for key, file_list in experiments.items():
    metrics_accumulator = defaultdict(list)

    for fname in file_list:
        file_path = os.path.join(input_dir, fname)
        with open(file_path, 'rb') as f:
            result = pickle.load(f)
        for k, v in result.items():
            metrics_accumulator[k].append(v)

    averaged_result = {k: float(np.mean(v)) for k, v in metrics_accumulator.items()}

    dataset, backbone, method, ood_set, correlation, seed = key
    output_filename = f"{dataset}^{backbone}^{method}^{ood_set}^{correlation}^{seed}^default.pkl"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'wb') as f:
        pickle.dump(averaged_result, f)

    print(f"Saved averaged result: {output_filename}")
