import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
input_dir = "pickles"


dataset_name = 'waterbirds' 
backbone_filter = 'resnet_18'
ood = 'placesbg'
correlation_filter = 'r50'

# dataset_name = 'celeba_blond' 
# backbone_filter = 'resnet_50'
# ood = 'farood'
# correlation_filter = 'r0.5'

output_dir = f"fintuned_bar_plots_{ood}"
os.makedirs(output_dir, exist_ok=True)

flags = ['default', 'finetuned']
metric = 'AUROC'

# Categories and color coding
feature_based = ["sprod3", "she", "knn", "rmds", "mds"]
hybrid_spaces = ["vim", "react"]
grad_based = ["gradnorm"]
output_based = ["klm", "mls", "ebo", "msp"]


all_methods = feature_based + hybrid_spaces + grad_based + output_based

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

all_methods_names = [name_dict[key] for key in all_methods]

method_colors = {}
for method in all_methods:
    if method in feature_based:
        method_colors[name_dict[method]] = 'darkred'
    elif method in hybrid_spaces:
        method_colors[name_dict[method]] = 'darkgreen'
    elif method in output_based:
        method_colors[name_dict[method]] = 'darkblue'
    elif method in grad_based:
        method_colors[name_dict[method]] = 'black'

# Collect records for both flags
records = []

for filename in os.listdir(input_dir):
    if not filename.endswith(".pkl") or '^' not in filename:
        continue

    parts = filename[:-4].split('^')
    if len(parts) < 7:
        continue

    dataset, backbone, method, ood_set, correlation, seed = parts[:6]
    flag = "_".join(parts[6:])

    if (dataset != dataset_name or
        backbone != backbone_filter or
        ood_set != ood or
        correlation != correlation_filter or
        flag not in flags):
        continue

    filepath = os.path.join(input_dir, filename)
    with open(filepath, 'rb') as f:
        result = pickle.load(f)

    records.append({
        "method": method,
        "flag": flag,
        "seed": seed,
        metric: result.get(metric, None)
    })

# Convert to DataFrame
df = pd.DataFrame.from_records(records)

if df.empty:
    print("⚠️ No data found for the specified setting.")
else:
    df_stats = df.groupby(['method', 'flag'])[metric].agg(['mean', 'std']).reset_index()

    # Ensure all methods are present for both flags
    data_plot = []
    for method in all_methods:
        for flag in flags:
            row = df_stats[(df_stats['method'] == method) & (df_stats['flag'] == flag)]
            if not row.empty:
                data_plot.append({
                    "method": method,
                    "flag": flag,
                    "mean": row['mean'].values[0],
                    "std": row['std'].values[0]
                })
            else:
                data_plot.append({
                    "method": method,
                    "flag": flag,
                    "mean": 0,
                    "std": 0
                })

    df_plot = pd.DataFrame(data_plot)

    # Build color palette per method
    palette = sns.color_palette("tab20", len(all_methods))
    method_bar_colors = dict(zip(all_methods, palette))
    
    # Plotting
    plt.figure(figsize=(4.5, 5))
    bar_width = 0.37
    x = list(range(len(all_methods)))
    
    for idx, method in enumerate(all_methods):
        for offset_idx, flag in enumerate(flags):
            row = df_plot[(df_plot['method'] == method) & (df_plot['flag'] == flag)]
            if row.empty:
                continue
    
            mean_val = row['mean'].values[0]
            std_val = row['std'].values[0]
    
            color = method_bar_colors[method]
            hatch = '//' if flag == 'finetuned' else None
            offset = -bar_width/2 if flag == 'default' else bar_width/2
    
            label = f"{method} ({flag})" if idx == 0 else None  # only label once per type
    
            plt.bar(
                x[idx] + offset,
                mean_val,
                yerr=std_val,
                width=bar_width,
                color=color,
                edgecolor='black',
                hatch=hatch,
                capsize=4,
                label=label
            )
    
    plt.xticks(ticks=x, labels=all_methods_names, rotation=55, ha='right', fontsize=15)
    plt.ylabel(f"{metric} (mean ± std)", fontsize=12)
    plt.title(f"$Spurious Correation: \\bf{{{correlation_filter[1:]}\\%}}$", fontsize=14)

    
    visible_means = df_plot[df_plot['mean'] > 0]
    if not visible_means.empty:
        ymin = max(0, (visible_means['mean'] - visible_means['std']).min() * 0.85)
        ymax = min(100, (visible_means['mean'] + visible_means['std']).max() * 1.02)
    else:
        ymin, ymax = 0, 1

    plt.ylim(39, 101)

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    ax = plt.gca()
    for label in ax.get_xticklabels():
        method = label.get_text()
        if method in feature_based:
            label.set_color("red")
        elif method in hybrid_spaces:
            label.set_color("green")
        elif method in output_based:
            label.set_color("blue")
        else:
            label.set_color("black")


    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(['Pretrained', 'Finetuned'], handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=13)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"finetuned_vs_pretrained_{dataset_name}_{backbone_filter}_{correlation_filter}.png")
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"✅ Saved fixed plot: {save_path}")
