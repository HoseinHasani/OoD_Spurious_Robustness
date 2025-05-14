import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Directories
input_dir = "pickles"
output_dir = "sprod_dataset_plots"
os.makedirs(output_dir, exist_ok=True)

# Parameters
metric = "AUROC"
flag_filter = "default"
backbone_filter = "resnet_50"

# Dataset metadata
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

for dataset_name in dataset_list:
    ood = near_ood_dataset[dataset_list.index(dataset_name)]
    correlation_filter = f"r{hard_correlations_dict[dataset_name]}"

    records = []

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

    df = pd.DataFrame(records)

    if df.empty:
        print(f"⚠️ No data found for {dataset_name} at correlation {correlation_filter}")
        continue

    df_stats = df.groupby(['method'])[metric].agg(['mean', 'std']).reset_index()

    # Complete with zeroes for missing methods
    df_complete = []
    for method in sprod_methods:
        row = df_stats[df_stats['method'] == method]
        if row.empty:
            df_complete.append({'method': method, 'mean': 0, 'std': 0})
        else:
            df_complete.append({
                'method': method,
                'mean': row['mean'].values[0],
                'std': row['std'].values[0]
            })
    df_plot = pd.DataFrame(df_complete)

    # Bar colors
    bar_colors = [method_colors[m] for m in df_plot['method']]

    # Plotting
    plt.figure(figsize=(5, 6))
    bars = plt.bar(
        df_plot['method'],
        df_plot['mean'],
        yerr=df_plot['std'],
        capsize=5,
        color=bar_colors,
        edgecolor='black'
    )

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    clean_name = standard_data_name[dataset_name]
    plt.title(f"{metric} for {clean_name}", fontsize=14)
    plt.xlabel("Method", fontsize=12)
    plt.ylabel(f"{metric} (mean ± std)", fontsize=12)

    # Dynamic ylim
    visible = df_plot[df_plot['mean'] > 0]
    if not visible.empty:
        ymin = max(0, (visible['mean'] - visible['std']).min() * 0.96)
        ymax = min(100, (visible['mean'] + visible['std']).max() * 1.01)
    else:
        ymin, ymax = 0, 1
    plt.ylim(ymin, ymax)

    plt.xticks(ticks=range(len(sprod_methods)), labels=sprod_methods, rotation=45, ha='right', fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"sprod_{dataset_name}.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"✅ Saved plot: {out_path}")
