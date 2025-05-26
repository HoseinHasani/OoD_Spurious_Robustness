import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# near_ood_dataset = ['SVHN', 'SVHN', 'SVHN', 'SVHN', 'SVHN']
near_ood_dataset = ['animals_ood', 'clbood', 'spurious_imagenet', 'urbn_no_car_ood', 'placesbg']

dataset = ['animals_metacoco', 'celeba_blond', 'spurious_imagenet', 'urbancars', 'waterbirds']

hard_correlations_dict ={
        'waterbirds': 90,
        'celeba_blond': 0.9,
        'urbancars': 95,
        'animals_metacoco': 95,
        'spurious_imagenet': 95
        }

select = 3

dataset_name = dataset[select]
corr = hard_correlations_dict[dataset_name]

directory = "pickles"

metric = "AUROC"
filters = {
    "dataset": dataset_name,
    "ood_set": near_ood_dataset[select],
    "correlation": f"r{corr}",
    "flag": "default",  
}

fields_to_compare = ["method", "backbone"]
fields_for_average = ["seed"]

records = []

for filename in os.listdir(directory):
    if not filename.endswith(".pkl") or '^' not in filename:
        continue

    parts = filename[:-4].split('^')
    if len(parts) < 7:
        continue

    dataset, backbone, method, ood_set, correlation, seed = parts[:6]
    flag = "_".join(parts[6:])

    if int(seed[1:]) < 20:
        continue

    if int(seed[1:]) > 25 and int(seed[1:]) < 100: 
        continue    
    
    if (
        dataset != filters["dataset"] or
        ood_set != filters["ood_set"] or
        correlation != filters["correlation"] or
        flag != filters["flag"]
    ):
        continue

    file_path = os.path.join(directory, filename)
    with open(file_path, 'rb') as f:
        result = pickle.load(f)

    record = {
        "dataset": dataset,
        "backbone": backbone,
        "method": method,
        "ood_set": ood_set,
        "correlation": correlation,
        "seed": seed,
        "flag": flag,
    }
    record.update(result)
    records.append(record)

df = pd.DataFrame.from_records(records)

group_keys = fields_to_compare.copy()
df_avg = df.groupby(group_keys)[metric].mean().reset_index()

pivot = df_avg.pivot(index=fields_to_compare[0], columns=fields_to_compare[1], values=metric)
os.makedirs('csvs', exist_ok=True)
output_csv_path = f"csvs/results_{dataset_name}_fpr.csv"
pivot.round(2).to_csv(output_csv_path)
print(f"Saved pivot table to {output_csv_path}")

# sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
# plt.title(f"{metric} across {fields_to_compare}")
# plt.show()
