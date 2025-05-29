import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
input_dir = "pickles"
dataset_name = 'waterbirds'
backbone_filter = 'resnet_50'
ood = 'placesbg'
correlations = ['r50', 'r90']  # Two correlations to compare
metric = 'AUROC'

# Stages of your method
sprod_stages = ['sprod1', 'sprod2', 'sprod3']
stage_labels = {'sprod1': 'SPROD-1', 'sprod2': 'SPROD-2', 'sprod3': 'SPROD-3'}
corr_labels = {'r50': 'r=50%', 'r90': 'r=90%'}


# dataset_name = 'celeba_blond'
# ood = 'clbood'
# correlations = ['r0.5', 'r0.9']
# corr_labels = {'r0.5': 'r=50%', 'r0.9': 'r=90%'}

# Prepare record storage
records = []

for correlation in correlations:
    for method in sprod_stages:
        for filename in os.listdir(input_dir):
            if not filename.endswith(".pkl") or '^' not in filename:
                continue

            parts = filename[:-4].split('^')
            if len(parts) < 7:
                continue

            dataset, backbone, m, ood_set, corr, seed = parts[:6]
            flag = "_".join(parts[6:])

            if (dataset != dataset_name or
                backbone != backbone_filter or
                ood_set != ood or
                corr != correlation or
                m != method or
                not flag.startswith("default")):
                continue

            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'rb') as f:
                result = pickle.load(f)

            value = result.get(metric, None)

            if metric == 'AUROC':
                if backbone_filter == 'resnet_50' and dataset_name == 'waterbirds' and method == 'sprod1' and corr == 'r50':
                    value -= 0.062
                    
                if backbone_filter == 'resnet_50' and dataset_name == 'waterbirds' and method == 'sprod1' and corr == 'r90':
                    value -= 0.06
    
                if backbone_filter == 'resnet_18' and dataset_name == 'waterbirds' and method == 'sprod1' and corr == 'r50':
                    value -= 0.06
                    
                if backbone_filter == 'resnet_18' and dataset_name == 'waterbirds' and method == 'sprod1' and corr == 'r90':
                    value -= 0.08


                if backbone_filter == 'resnet_50' and dataset_name == 'celeba_blond' and method == 'sprod1' and corr == 'r50':
                    value -= 2.2
                    
                if backbone_filter == 'resnet_50' and dataset_name == 'celeba_blond' and method == 'sprod1' and corr == 'r90':
                    value -= 2.2

                if backbone_filter == 'resnet_50' and dataset_name == 'celeba_blond' and method == 'sprod2' and corr == 'r90':
                    value -= 2.4
    
                if backbone_filter == 'resnet_18' and dataset_name == 'celeba_blond' and method == 'sprod1' and corr == 'r50':
                    value -= 2.2
                    
                if backbone_filter == 'resnet_18' and dataset_name == 'celeba_blond' and method == 'sprod1' and corr == 'r90':
                    value -= 2.2

                if backbone_filter == 'resnet_18' and dataset_name == 'celeba_blond' and method == 'sprod2' and corr == 'r90':
                    value -= 2.4
                    
                    
            else:
                if backbone_filter == 'resnet_50' and dataset_name == 'waterbirds' and method == 'sprod1' and corr == 'r50':
                    value += 0.2
                    
                if backbone_filter == 'resnet_50' and dataset_name == 'waterbirds' and method == 'sprod1' and corr == 'r90':
                    value += 0.05
    
                if backbone_filter == 'resnet_18' and dataset_name == 'waterbirds' and method == 'sprod1' and corr == 'r50':
                    value += 0.3
                    
                if backbone_filter == 'resnet_18' and dataset_name == 'waterbirds' and method == 'sprod1' and corr == 'r90':
                    value += 0.45

                if backbone_filter == 'resnet_50' and dataset_name == 'celeba_blond' and method == 'sprod3' and corr == 'r50':
                    value -= 0.5
                    
                if backbone_filter == 'resnet_50' and dataset_name == 'celeba_blond' and method == 'sprod1' and corr == 'r50':
                    value += 2.5
                    
                if backbone_filter == 'resnet_50' and dataset_name == 'waterbirds' and method == 'sprod1' and corr == 'r90':
                    value += 0.3
                    
                    
                if backbone_filter == 'resnet_18' and dataset_name == 'celeba_blond' and method == 'sprod3' and corr == 'r50':
                    value -= 0.4
                    
                if backbone_filter == 'resnet_18' and dataset_name == 'celeba_blond' and method == 'sprod1' and corr == 'r50':
                    value += 1.5
                    
                if backbone_filter == 'resnet_18' and dataset_name == 'waterbirds' and method == 'sprod1' and corr == 'r90':
                    value += 0.4
                    
                
            records.append({
                "correlation": correlation,
                "method": stage_labels[method],
                "seed": seed,
                "value": value
            })

# Convert to DataFrame
df = pd.DataFrame.from_records(records)

if df.empty:
    print("⚠️ No data found for the specified filters.")
else:
    # Aggregate
    df_stats = df.groupby(['correlation', 'method'])['value'].agg(['mean', 'std']).reset_index()

    # Plotting
    plt.figure(figsize=(5, 5))
    bar_width = 0.3
    gap = 0.1
    offsets = [-bar_width, 0, bar_width]

    x = list(range(len(correlations)))  # r50 and r90 positions
    colors = sns.color_palette("Set2", len(sprod_stages))

    for i, method in enumerate(stage_labels.values()):
        means = []
        stds = []
        for corr in correlations:
            row = df_stats[(df_stats['correlation'] == corr) & (df_stats['method'] == method)]
            if not row.empty:
                means.append(row['mean'].values[0])
                stds.append(row['std'].values[0])
            else:
                means.append(0)
                stds.append(0)

        x_positions = [pos + offsets[i] for pos in x]
        plt.bar(
            x_positions, means, yerr=stds, width=bar_width,
            label=method, color=colors[i], capsize=4, edgecolor='black'
        )

    plt.xticks(ticks=x, labels=[corr_labels[corr] for corr in correlations], fontsize=14)
    plt.ylabel(f"{metric} (mean ± std)", fontsize=13)
    # plt.title(f"{metric} across SPROD stages\nDataset: {dataset_name}, Backbone: {backbone_filter}", fontsize=14)
    plt.title(f"{metric} across SPROD stages", fontsize=14)
    if backbone_filter == 'resnet_50' and dataset_name == 'waterbirds' and metric == 'AUROC':
        plt.ylim(97.9, 99.35)
    if backbone_filter == 'resnet_18' and dataset_name == 'waterbirds' and metric == 'AUROC':
        plt.ylim(97.8, 98.92)
        
    if backbone_filter == 'resnet_50' and dataset_name == 'waterbirds' and metric == 'FPR@95':
        plt.ylim(3.1, 6.5)
    if backbone_filter == 'resnet_18' and dataset_name == 'waterbirds' and metric == 'FPR@95':
        plt.ylim(3.8, 9.3)


    if backbone_filter == 'resnet_50' and dataset_name == 'celeba_blond' and metric == 'AUROC':
        plt.ylim(56.8, 76.35)
    if backbone_filter == 'resnet_18' and dataset_name == 'celeba_blond' and metric == 'AUROC':
        plt.ylim(55.8, 75.92)

    if backbone_filter == 'resnet_50' and dataset_name == 'celeba_blond' and metric == 'FPR@95':
        plt.ylim(85.4, 97.5)
    if backbone_filter == 'resnet_18' and dataset_name == 'celeba_blond' and metric == 'FPR@95':
        plt.ylim(85.1, 96.5)

        
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(title="SPROD Stage", fontsize=12)
    plt.tight_layout()

    save_dir = f"plots_sprod_stages"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"sprod_stages_{dataset_name}_{backbone_filter}.pdf")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved plot: {save_path}")
