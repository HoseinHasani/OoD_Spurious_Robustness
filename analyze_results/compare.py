import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

directory = "pickles"

metric = "AUROC"
filters = {
    "dataset": "spurious_imagenet",
    "ood_set": "spurious_imagenet",
    "correlation": "r95",
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

output_csv_path = f"results_{metric.lower()}_pivot.csv"
pivot.round(2).to_csv(output_csv_path)
print(f"Saved pivot table to {output_csv_path}")

sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
plt.title(f"{metric} across {fields_to_compare}")
plt.show()
