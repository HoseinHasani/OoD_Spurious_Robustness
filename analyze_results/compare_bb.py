import os
import pickle
import pandas as pd

# Define constants
directory = "pickles"
target_method = "sprod3"
metric = "FPR@95"

datasets = ['animals_metacoco', 'celeba_blond', 'spurious_imagenet', 'urbancars', 'waterbirds']

hard_correlations_dict = {
    'waterbirds': 90,
    'celeba_blond': 0.9,
    'urbancars': 95,
    'animals_metacoco': 95,
    'spurious_imagenet': 95
}

near_ood_dataset = {
    'animals_metacoco': 'animals_ood',
    'celeba_blond': 'clbood',
    'spurious_imagenet': 'spurious_imagenet',
    'urbancars': 'urbn_no_car_ood',
    'waterbirds': 'placesbg'
}

records = []

# Step 1: Load and filter all valid records
for filename in os.listdir(directory):
    if not filename.endswith(".pkl") or '^' not in filename:
        continue

    parts = filename[:-4].split('^')
    if len(parts) < 7:
        continue

    dataset, backbone, method, ood_set, correlation, seed = parts[:6]
    flag = "^".join(parts[6:])
    
    if "sprod1" in method or "sprod2" in method or "sprod4" in method:
        continue

    if dataset not in datasets:
        continue

    if (
        ood_set != near_ood_dataset[dataset] or
        correlation != f"r{hard_correlations_dict[dataset]}" or
        flag != "default"
    ):
        continue

    file_path = os.path.join(directory, filename)
    try:
        with open(file_path, 'rb') as f:
            result = pickle.load(f)
    except Exception as e:
        print(f"Failed to read {filename}: {e}")
        continue

    print(filename)
    
    records.append({
        "dataset": dataset,
        "backbone": backbone,
        "method": method,
        "seed": seed,
        metric: result.get(metric, None)
    })

# Step 2: Create dataframe and compute averages over seeds
df = pd.DataFrame(records)
df = df.dropna(subset=[metric])
df_avg = df.groupby(["dataset", "backbone", "method"])[metric].mean().reset_index()

# Step 3: Compute ranks of all methods per (dataset, backbone)
df_avg["rank"] = df_avg.groupby(["dataset", "backbone"])[metric].rank(ascending=True, method="min")

# Step 4: Extract only sprod3's ranks
sprod3_ranks = df_avg[df_avg["method"] == target_method]

# Step 5: Pivot to get datasets as rows and backbones as columns
pivot = sprod3_ranks.pivot(index="dataset", columns="backbone", values="rank")
pivot = pivot.round(0).astype("Int64")  # optional: round and make integer-looking

# Step 6: Save to CSV
os.makedirs("csvs", exist_ok=True)
pivot.to_csv("csvs/sprod3_ranks.csv")
print("Saved sprod3 rank table to csvs/sprod3_ranks.csv")
