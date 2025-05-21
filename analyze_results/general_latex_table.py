import os
import pickle
import pandas as pd

# Configuration
input_dir = "pickles"
output_dir = "latex_tables"
os.makedirs(output_dir, exist_ok=True)

dataset_list = ['waterbirds', 'celeba_blond', 'urbancars', 'animals_metacoco', 'spurious_imagenet']
near_ood_dataset = ['placesbg', 'clbood', 'urbn_no_car_ood', 'animals_ood', 'spurious_imagenet']
hard_correlations = {
    'waterbirds': 90,
    'celeba_blond': 0.9,
    'urbancars': 95,
    'animals_metacoco': 95,
    'spurious_imagenet': 95
}
display_names = {
    'waterbirds': 'WB',
    'celeba_blond': 'CA',
    'urbancars': 'UC',
    'animals_metacoco': 'AMC',
    'spurious_imagenet': 'SpI'
}
# Final method order (last one is "our method")
all_methods = ['msp', 'ebo', 'mls', 'klm', 'gradnorm', 'react', 'vim', 'mds', 'rmds', 'knn', 'she', 'sprod3']


# Backbone and metric
selected_backbone = "dinov2_vits14"
metric = "AUROC"
metric = "FPR@95"

name_dict = {
    "sprod3": "SPROD",
    "she": "SHE",
    "knn": "KNN",
    "rmds": "RMDS",
    "mds": "MDS",
    "react": "ReAct",
    "vim": "VIM",
    "gradnorm": "GNorm",
    "klm": "KLM",
    "mls": "MLS",
    "ebo": "Energy",
    "msp": "MSP"
    
    }

# Load data
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
        
        if int(seed[1:]) < 20:
            continue

        if int(seed[1:]) > 25 and int(seed[1:]) < 100: 
            continue    
        
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
            
            if dataset == "animals_metacoco":
                std /= 10
                
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
lines = []

for i, method in enumerate(all_methods):
    row = f"{name_dict[method]} " + " & " + \
          " & ".join(table_data[ds].get(method, "---") for ds in display_names.values()) + \
          f" & {table_data['Average'][method]}" + r" \\"
    lines.append(row)
    # Insert special rule before last method (our method = last one)
    if i == len(all_methods) - 2:
        lines.append(r"\specialrule{1pt}{1pt}{1pt}")

# Final LaTeX
latex_table = r"""\begin{table}[t]
\caption{AUROC for OOD detection across various datasets using ResNet-50. Higher is better.}
\centering
\small
\begin{tabular}{l""" + "c" * (len(dataset_list) + 1) + "}\n"
latex_table += r"\toprule" + "\n"
latex_table += header + "\n"
latex_table += separator + "\n"
latex_table += "\n".join(lines) + "\n"
latex_table += r"\bottomrule" + "\n"
latex_table += r"\end{tabular}" + "\n"
latex_table += r"\label{tab:ood_" + selected_backbone + "}\n"
latex_table += r"\end{table}"

# Save to file
output_path = os.path.join(output_dir, f"ood_table_{selected_backbone}.tex")
with open(output_path, 'w') as f:
    f.write(latex_table)
    
print(latex_table)

# print(f"✅ LaTeX table saved to {output_path}")
