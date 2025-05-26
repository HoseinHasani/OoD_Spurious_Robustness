import os
import pickle
import pandas as pd
from collections import defaultdict

# Config
input_dir = "pickles"
output_dir = "latex_tables"
os.makedirs(output_dir, exist_ok=True)

metric = "AUROC"
# metric = "FPR@95"

flag_filter = "default"
target_datasets = ['waterbirds', 'celeba_blond', 'urbancars', 'animals_metacoco', 'spurious_imagenet']
near_ood_map = {
    'waterbirds': 'placesbg',
    'celeba_blond': 'clbood',
    'urbancars': 'urbn_no_car_ood',
    'animals_metacoco': 'animals_ood',
    'spurious_imagenet': 'spurious_imagenet'
}
hard_correlations = {
    'waterbirds': 90,
    'celeba_blond': 0.9,
    'urbancars': 95,
    'animals_metacoco': 95,
    'spurious_imagenet': 95
}

backbones = ['resnet_18', 'resnet_34', 'resnet_50', 'resnet_101', 'dinov2_vits14', 
             'ViT_S', 'Swin_B', 'DeiT_B', 'ConvNeXt_B', 'BiT_M_R50x1']

all_methods = ['msp', 'ebo', 'mls', 'klm', 'gradnorm', 'react', 'vim', 'mds', 'rmds', 'knn', 'she', 'sprod3']

method_display = {
    "sprod3": "SPROD", "she": "SHE", "knn": "KNN", "rmds": "RMDS", "mds": "MDS", "react": "ReAct", "vim": "VIM",
    "gradnorm": "GNorm", "klm": "KLM", "mls": "MLS", "ebo": "Energy", "msp": "MSP"
}

backbone_display = {
    "resnet_18": "R18", "resnet_34": "R34", "resnet_50": "R50", "resnet_101": "R101",
    "dinov2_vits14": "DINOv2", "ViT_S": "ViT", "DeiT_B": "DeiT", "BiT_M_R50x1": "BiT",
    "ConvNeXt_B": "CvNxt", "Swin_B": "Swin",
}

backbone_display_names = [backbone_display[name] for name in backbones]

# Step 1: Load results
records = []
for fname in os.listdir(input_dir):
    if not fname.endswith('.pkl') or '^' not in fname:
        continue
    parts = fname[:-4].split('^')
    if len(parts) < 7:
        continue
    ds, backbone, method, ood_set, corr, seed = parts[:6]
    flag = "^".join(parts[6:])
    
    if int(seed[1:]) < 20 or (25 < int(seed[1:]) < 100):
        continue
    
    if ds not in target_datasets:
        continue
    if ood_set != near_ood_map[ds] or corr != f"r{hard_correlations[ds]}" or flag != flag_filter:
        continue

    try:
        with open(os.path.join(input_dir, fname), 'rb') as f:
            result = pickle.load(f)
    except Exception as e:
        print(f"Could not read {fname}: {e}")
        continue
    
    records.append({
        "dataset": ds,
        "backbone": backbone,
        "method": method,
        "seed": seed,
        metric: result.get(metric, None)
    })

# Step 2: Compute average per method-backbone
df = pd.DataFrame(records)
df = df.dropna(subset=[metric])
grouped = df.groupby(["method", "backbone"])[metric].mean().reset_index()

# Step 3: Pivot and compute average column
pivot = grouped.pivot(index="method", columns="backbone", values=metric)
pivot = pivot.reindex(all_methods)  # Ensure consistent method order
pivot = pivot[backbones]  # Ensure consistent backbone order
pivot["Average"] = pivot.mean(axis=1)

# Step 4: Format for LaTeX
def format_val(val):
    return f"{val:.1f}" if pd.notnull(val) else "---"

latex_rows = []
for method in pivot.index:
    row = [method_display.get(method, method)]
    for val in pivot.loc[method]:
        row.append(format_val(val))
    latex_rows.append(row)

# Step 5: Generate LaTeX table
header = "Method & " + " & ".join(backbone_display_names) + " & Avg. \\\\"
separator = "\\midrule"
lines = [" & ".join(row) + r" \\" for row in latex_rows[:-1]]
lines.append(r"\midrule")
lines.append(" & ".join(latex_rows[-1]) + r" \\")

latex_table = r"""\begin{table}[t]
\caption{""" + f"{metric} averaged across datasets for each backbone and method, including overall average." + r"""}
\centering
\small
\begin{tabular}{l""" + "c" * (len(backbones) + 1) + "}\n"
latex_table += r"\toprule" + "\n"
latex_table += header + "\n"
latex_table += separator + "\n"
latex_table += "\n".join(lines) + "\n"
latex_table += r"\bottomrule" + "\n"
latex_table += r"\end{tabular}" + "\n"
latex_table += r"\label{tab:ood_backbone_avg}" + "\n"
latex_table += r"\end{table}"

# Save LaTeX
output_path = os.path.join(output_dir, f"ood_table_backbones_avg_with_overall_{metric}.tex")
with open(output_path, 'w') as f:
    f.write(latex_table)

print("\n" + latex_table)
