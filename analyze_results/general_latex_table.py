import os
import pickle
import pandas as pd

# Configuration
input_dir = "pickles"
output_dir = "latex_tables"
os.makedirs(output_dir, exist_ok=True)

dataset_list = ['animals_metacoco', 'celeba_blond', 'spurious_imagenet', 'urbancars', 'waterbirds']
near_ood_dataset = ['animals_ood', 'clbood', 'spurious_imagenet', 'urbn_no_car_ood', 'placesbg']
hard_correlations = {
    'waterbirds': 90,
    'celeba_blond': 0.9,
    'urbancars': 95,
    'animals_metacoco': 95,
    'spurious_imagenet': 95
}
display_names = {
    'waterbirds': 'Waterbirds',
    'celeba_blond': 'CelebA',
    'urbancars': 'UrbanCars',
    'animals_metacoco': 'AnimalsMetaCoCo',
    'spurious_imagenet': 'SpuriousImageNet'
}
feature_methods = ['sprod3', 'knn', 'mds', 'rmds', 'she']
output_methods = ['msp', 'mls', 'gradnorm', 'ebo', 'vim']
all_methods = feature_methods + output_methods
all_methods = ['msp', 'mds', 'rmds', 'ebo', 'gradnorm', 'react', 'mls', 'klm', 'knn', 'she', 'vim', 'sprod3']

# Choose backbone
selected_backbone = "resnet_50"  # Change as needed
metric = "AUROC"

# Load all records first
records = []

for dataset, ood in zip(dataset_list, near_ood_dataset):
    corr_tag = f"r{hard_correlations[dataset]}"
    for fname in os.listdir(input_dir):
        if not fname.endswith('.pkl') or '^' not in fname:
            continue
        parts = fname[:-4].split('^')
        if len(parts) < 7:
            continue
        ds, backbone, method, ood_set, corr, seed = parts[:6]
        flag = "_".join(parts[6:])
        if ds != dataset or ood_set != ood or corr != corr_tag or flag != "default":
            continue
        if backbone != selected_backbone:
            continue

        with open(os.path.join(input_dir, fname), 'rb') as f:
            result = pickle.load(f)
        records.append({
            "dataset": ds,
            "method": method,
            "seed": seed,
            metric: result.get(metric, None)
        })

# Build dataframe
df = pd.DataFrame(records)
if df.empty:
    raise ValueError(f"No data found for backbone '{selected_backbone}'.")

# Compute mean/std
stats = df.groupby(['dataset', 'method'])[metric].agg(['mean', 'std']).reset_index()

# Build LaTeX table data
table_data = {}

for dataset in dataset_list:
    readable_name = display_names[dataset]
    sub = stats[stats['dataset'] == dataset]
    row = {}
    for method in all_methods:
        entry = sub[sub['method'] == method]
        if not entry.empty:
            mean = entry['mean'].values[0]
            std = entry['std'].values[0]
            row[method] = f"${mean:.1f}_{{\\textcolor{{gray}}{{\\pm{std:.1f}}}}}$"
        else:
            row[method] = "---"
    table_data[readable_name] = row

# Compute average row
avg_row = {}
for method in all_methods:
    vals = []
    for dataset in display_names.values():
        entry = table_data[dataset][method]
        if entry != "---":
            try:
                mean_val = float(entry.split("_")[0].replace("$", "").strip())
                vals.append(mean_val)
            except:
                continue
    avg_row[method] = f"${sum(vals)/len(vals):.1f}$" if vals else "---"
table_data["Average"] = avg_row

# Generate LaTeX table string
header = "Method & " + " & ".join(display_names[ds] for ds in dataset_list) + " & Average \\\\"
separator = "\\midrule"
lines = [f"{method} " +
         " & " + " & ".join(table_data[ds].get(method, "---") for ds in display_names.values()) +
         f" & {table_data['Average'][method]}" + r" \\"
         for method in all_methods]

latex_table = r"""\begin{table}[t]
\centering
\small
\begin{tabular}{l""" + "c" * (len(dataset_list) + 1) + "}\n"
latex_table += r"\toprule" + "\n"
latex_table += header + "\n"
latex_table += separator + "\n"
latex_table += "\n".join(lines) + "\n"
latex_table += r"\bottomrule" + "\n"
latex_table += r"\end{tabular}" + "\n"
latex_table += f"\\caption{{{metric}}}" + "\n"
latex_table += r"\label{tab:ood_" + selected_backbone + "}\n"
latex_table += r"\end{table}"

# Print and save
print(latex_table)
output_path = os.path.join(output_dir, f"ood_table_{selected_backbone}.tex")
with open(output_path, 'w') as f:
    f.write(latex_table)
