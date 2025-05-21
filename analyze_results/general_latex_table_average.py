import os
import pickle
import pandas as pd
from collections import defaultdict

# Config
input_dir = "pickles"
output_dir = "latex_tables"
os.makedirs(output_dir, exist_ok=True)

metric = "AUROC"
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
backbones = ['BiT_M_R101x1', 'BiT_M_R50x1', 'BiT_M_R50x3', 'DeiT_B', 'DeiT_S',
             'DeiT_Ti', 'ViT_B', 'ViT_S', 'ViT_Ti', 'clip_RN50', 'clip_ViT_B_16',
             'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vits14', 'resnet_101',
             'resnet_18', 'resnet_50']

backbones = ['DeiT_B', 'ViT_S', 'dinov2_vits14', 'resnet_101',
             'resnet_18', 'resnet_50']

all_methods = ['msp', 'ebo', 'mls', 'klm', 'gradnorm', 'react', 'vim', 'mds', 'rmds', 'knn', 'she', 'sprod3']
method_display = {
    "sprod3": "SPROD", "she": "SHE", "knn": "KNN", "rmds": "RMDS", "mds": "MDS", "react": "ReAct", "vim": "VIM",
    "gradnorm": "GNorm", "klm": "KLM", "mls": "MLS", "ebo": "Energy", "msp": "MSP"
}

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

# Step 3: Pivot for LaTeX table
pivot = grouped.pivot(index="method", columns="backbone", values=metric)
pivot = pivot.reindex(all_methods)  # Ensure order of methods
pivot = pivot[backbones]  # Ensure order of backbones

# Step 4: Format LaTeX table
def format_val(val):
    return f"{val:.1f}" if pd.notnull(val) else "---"

def bold_max(series):
    max_val = series.max()
    return [f"\\textbf{{{format_val(v)}}}" if v == max_val else format_val(v) for v in series]

# Optional: bold best methods per column
latex_rows = []
for method in pivot.index:
    row = [method_display.get(method, method)]
    for val in pivot.loc[method]:
        row.append(format_val(val))
    latex_rows.append(row)

# Alternatively: bold top per column
# pivot_formatted = pivot.apply(bold_max, axis=0)
# latex_rows = [[method_display.get(method, method)] + list(pivot_formatted.loc[method]) for method in pivot.index]

# Step 5: Assemble LaTeX
header = "Method & " + " & ".join(backbones) + " \\\\"
separator = "\\midrule"
lines = [" & ".join(row) + r" \\" for row in latex_rows]

latex_table = r"""\begin{table}[t]
\caption{""" + f"{metric} averaged across datasets for each backbone and method." + r""" Higher is better.}
\centering
\small
\begin{tabular}{l""" + "c" * len(backbones) + "}\n"
latex_table += r"\toprule" + "\n"
latex_table += header + "\n"
latex_table += separator + "\n"
latex_table += "\n".join(lines) + "\n"
latex_table += r"\bottomrule" + "\n"
latex_table += r"\end{tabular}" + "\n"
latex_table += r"\label{tab:ood_backbone_avg}" + "\n"
latex_table += r"\end{table}"

# Save
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, f"ood_table_backbones_avg_{metric}.tex"), 'w') as f:
    f.write(latex_table)
    
print(latex_table)

print("✅ LaTeX table generated and saved.")
