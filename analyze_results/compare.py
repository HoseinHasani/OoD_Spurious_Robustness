import os
import pickle
import pandas as pd
from collections import defaultdict

directory = "pickles"

records = []

for filename in os.listdir(directory):
    if '^' in filename and filename.endswith(".pkl"):
        parts = filename[:-4].split('^')
        if len(parts) < 7:
            continue
        dataset, backbone, method, ood_set, correlation, seed = parts[:6]
        flag = "_".join(parts[6:])

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

fixed_fields = ["dataset", "correlation", "ood_set"]
fields_to_compare = ["method", "backbone"]
fields_for_average = ["seed"]

metric = "AUROC"

filters = {
    "dataset": "celeba_blond",
    "ood_set": "clbood",
    # "backbone": "resnet_18",
    "correlation": "r0.9",
    "flag": "default"
}
for key, value in filters.items():
    df = df[df[key] == value]

group_keys = fields_to_compare.copy()
df_avg = df.groupby(group_keys)[metric].mean().reset_index()

pivot = df_avg.pivot(index=fields_to_compare[0], columns=fields_to_compare[1], values=metric)
print(pivot.round(2))

output_csv_path = f"results_{metric.lower()}_pivot.csv"
pivot.round(2).to_csv(output_csv_path)

print(f"Saved pivot table to {output_csv_path}")

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
plt.title(f"{metric} across {fields_to_compare}")
plt.show()