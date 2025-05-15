import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Config
input_dir = "pickles"
output_path = "sprod_comparison_across_datasets.png"
metric = "AUROC"
flag_filter = "default"
backbone_filter = "resnet_50"

# Dataset mappings
dataset_list = ['animals_metacoco', 'celeba_blond', 'spurious_imagenet', 'urbancars', 'waterbirds']
near_ood_dataset = ['animals_ood', 'clbood', 'spurious_imagenet', 'urbn_no_car_ood', 'placesbg']
hard_correlations_dict = {
    'waterbirds': 90,
    'celeba_blond': 0.9,
    'urbancars': 95,
    'animals_metacoco': 95,
    'spurious_imagenet': 95
}
standard_data_name = {
    'waterbirds': 'Waterbirds',
    'celeba_blond': 'CelebA',
    'urbancars': 'UrbanCars',
    'animals_metacoco': 'AnimalsMetaCoCo',
    'spurious_imagenet': 'SpuriousImageNet'
}

# SProd methods only
sprod_methods = ['sprod1', 'sprod2', 'sprod3', 'sprod4']
palette = sns.color_palette("tab10", len(sprod_methods))
method_colors = dict(zip(sprod_methods, palette))

# Collect all records
records = []

for dataset_name in dataset_list:
    ood = near_ood_dataset[dataset_list.index(dataset_name)]
    correlation_filter = f"r{hard_correlations_dict[dataset_name]}"
    
    for filename in os.listdir(input_dir):
        if not filename.endswith(".pkl") or '^' not in filename:
            continue

        parts = filename[:-4].split('^')
        if len(parts) < 7:
            continue

        dataset, backbone, method, ood_set, correlation, seed = parts[:6]
        flag = "_".join(parts[6:])

        if (
            dataset != dataset_name or
            ood_set != ood or
            correlation != correlation_filter or
            method not in sprod_methods or
            flag != flag_filter
        ):
            continue

        filepath = os.path.join(input_dir, filename)
        with open(filepath, 'rb') as f:
            result = pickle.load(f)

        records.append({
            "dataset": standard_data_name[dataset],
            "method": method,
            "seed": seed,
            metric: result.get(metric, None)
        })

# Build dataframe
df = pd.DataFrame(records)

if df.empty:
    print("⚠️ No data found for SProd methods across datasets.")
else:
    df_stats = df.groupby(['dataset', 'method'])[metric].agg(['mean', 'std']).reset_index()

    # Bar plot
    plt.figure(figsize=(10, 6))
    bar_width = 0.2
    x_locs = list(range(len(dataset_list)))

    for idx, method in enumerate(sprod_methods):
        x_offset = [x + (idx - 1.5) * bar_width for x in x_locs]
        mean_vals, std_vals = [], []

        for ds in dataset_list:
            ds_std = standard_data_name[ds]
            row = df_stats[(df_stats['dataset'] == ds_std) & (df_stats['method'] == method)]
            if not row.empty:
                mean_vals.append(row['mean'].values[0])
                std_vals.append(row['std'].values[0])
            else:
                mean_vals.append(0)
                std_vals.append(0)

        plt.bar(
            x_offset,
            mean_vals,
            yerr=std_vals,
            capsize=4,
            width=bar_width,
            label=method,
            color=method_colors[method],
            edgecolor='black'
        )

    # Final plot formatting
    dataset_labels = [standard_data_name[ds] for ds in dataset_list]
    plt.xticks(ticks=x_locs, labels=dataset_labels, fontsize=12)
    plt.ylabel(f"{metric} (mean ± std)", fontsize=12)
    plt.title("SProd Comparison Across Datasets", fontsize=14)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(title="Method", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"✅ Saved SProd comparison plot: {output_path}")
