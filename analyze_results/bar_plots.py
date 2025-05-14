import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_dir = "pickles"
output_dir = "method_bar_plots"
os.makedirs(output_dir, exist_ok=True)

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

feature_based = ['sprod3', 'knn', 'mds', 'rmds', 'react', 'she', 'openmax']
output_based = ['msp', 'mls', 'gradnorm', 'ebo', 'vim', 'gradnorm']
all_methods = feature_based + output_based

# Use a distinct, consistent color per method
method_colors = dict(zip(all_methods, sns.color_palette("tab20", len(all_methods))))

metric = "AUROC"

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
            flag != "default"
        ):
            continue

        filepath = os.path.join(input_dir, filename)
        with open(filepath, 'rb') as f:
            result = pickle.load(f)

        record = {
            "dataset": dataset,
            "backbone": backbone,
            "method": method,
            "seed": seed,
            metric: result.get(metric, None)
        }
        records.append(record)

    df = pd.DataFrame.from_records(records)

    if df.empty:
        print(f"⚠️ No data found for {dataset_name} at correlation {correlation_filter}")
        continue

    df_stats = df.groupby(['backbone', 'method'])[metric].agg(['mean', 'std']).reset_index()

    for backbone in df_stats['backbone'].unique():
        df_backbone = df_stats[df_stats['backbone'] == backbone]

        # Include missing methods with zeros
        complete_df = []
        for method in all_methods:
            row = df_backbone[df_backbone['method'] == method]
            if row.empty:
                complete_df.append({'method': method, 'mean': 0, 'std': 0})
            else:
                complete_df.append({'method': method, 'mean': row['mean'].values[0], 'std': row['std'].values[0]})
        df_plot = pd.DataFrame(complete_df)

        # Use consistent, unique colors for each method
        colors = [method_colors[m] for m in df_plot['method']]

        plt.figure(figsize=(6, 6))  # reduced width
        bars = plt.bar(
            df_plot['method'],
            df_plot['mean'],
            yerr=df_plot['std'],
            capsize=5,
            color=colors,
            edgecolor='black'
        )

        plt.grid(axis='y', linestyle='--', alpha=0.3)

        clean_name = standard_data_name[dataset_name]
        plt.title(f"{metric} for {clean_name} - {backbone}", fontsize=14)
        plt.xlabel("OOD Detection Method", fontsize=12)
        plt.ylabel(f"{metric} (mean ± std)", fontsize=12)

        plt.xticks(ticks=range(len(all_methods)), labels=all_methods,
                   rotation=45, ha='right', fontsize=14)
        ax = plt.gca()
        for label in ax.get_xticklabels():
            method = label.get_text()
            label.set_color('darkred' if method in feature_based else 'darkblue')

        visible_means = df_plot[df_plot['mean'] > 0]
        if not visible_means.empty:
            ymin = max(0, (visible_means['mean'] - visible_means['std']).min() * 0.9)
            ymax = min(100, (visible_means['mean'] + visible_means['std']).max() * 1.02)
        else:
            ymin, ymax = 0, 1

        plt.ylim(ymin, ymax)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{dataset_name}_{backbone}.png")
        plt.savefig(save_path, dpi=160)
        plt.close()
        print(f"✅ Saved plot: {save_path}")
