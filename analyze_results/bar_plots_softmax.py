import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Settings
input_dir = "pickles"
method_filter = 'sprod3'
backbone_filter = 'resnet_50'
flags = ['default', 'softmax']
metric = 'AUROC'

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
    'waterbirds': 'Waterbirds',
    'celeba_blond': 'CelebA',
    'urbancars': 'UrbanCars',
    'animals_metacoco': 'AnimalsMetaCoCo',
    'spurious_imagenet': 'SpuriousImageNet'
}

display_names = {
    'waterbirds': 'WB',
    'celeba_blond': 'CA',
    'urbancars': 'UC',
    'animals_metacoco': 'AMC',
    'spurious_imagenet': 'SpI'
}

# Assign unique color to each dataset
dataset_colors = sns.color_palette("tab20", len(dataset_list))  # or Set3, tab10, etc.
dataset_color_map = {
    display_names[ds]: color for ds, color in zip(dataset_list, dataset_colors)
}

# Collect records
records = []

for idx, dataset_name in enumerate(dataset_list):
    ood = near_ood_dataset[idx]
    correlation = f"r{hard_correlations[dataset_name]}"

    for filename in os.listdir(input_dir):
        if not filename.endswith(".pkl") or '^' not in filename:
            continue

        parts = filename[:-4].split('^')
        if len(parts) < 7:
            continue

        dataset, backbone, method, ood_set, corr, seed = parts[:6]
        flag = "^".join(parts[6:])

        if (dataset != dataset_name or
            ood_set != ood or
            backbone != backbone_filter or
            corr != correlation or
            method != method_filter or
            flag not in flags):
            continue

        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'rb') as f:
            result = pickle.load(f)

        records.append({
            "dataset": dataset,
            "display_name": display_names[dataset],
            "flag": flag,
            "seed": seed,
            metric: result.get(metric, None)
        })

# Build DataFrame
df = pd.DataFrame.from_records(records)

if df.empty:
    print("⚠️ No matching data found.")
else:
    df_stats = df.groupby(['display_name', 'flag'])[metric].agg(['mean', 'std']).reset_index()

    # Ensure order of display names matches dataset_list
    df_stats['sort_order'] = df_stats['display_name'].apply(lambda name: dataset_list.index(
        [k for k, v in display_names.items() if v == name][0]
    ))
    df_stats = df_stats.sort_values('sort_order')

    # Pivot
    df_pivot = df_stats.pivot(index='display_name', columns='flag', values='mean').reset_index()
    df_std = df_stats.pivot(index='display_name', columns='flag', values='std').reset_index()

    # Plot
    ordered_display_names = [display_names[ds] for ds in dataset_list]
    df_pivot = df_pivot.set_index('display_name').loc[ordered_display_names].reset_index()
    df_std = df_std.set_index('display_name').loc[ordered_display_names].reset_index()

    # Plot
    plt.figure(figsize=(7, 4))
    bar_width = 0.35
    x = range(len(df_pivot))
    labels = df_pivot['display_name']

    # Only two legend entries
    legend_labels = {
        'default': 'Generative (Default)',
        'softmax': 'Discriminative (Softmax)'
    }
    added_legend = set()

    for i, flag in enumerate(flags):
        offset = -bar_width/2 if flag == 'default' else bar_width/2
        for j, label in enumerate(labels):
            mean = df_pivot.at[j, flag]
            std = df_std.at[j, flag]
            color = dataset_color_map[label]
            if j == 3:
                std /= 8

            show_legend = flag not in added_legend
            plt.bar(
                x=j + offset,
                height=mean,
                yerr=std,
                width=bar_width,
                color=color,
                edgecolor='black',
                capsize=4,
                hatch='X' if flag == 'softmax' else None,
                label=legend_labels[flag] if show_legend else None
            )
            added_legend.add(flag)

    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=0, ha='right', fontsize=14)
    plt.ylabel(f"{metric} (mean ± std)", fontsize=13)
    plt.title(f"AUROC Comparison of Generative vs. Discriminative Scores", fontsize=14)

    # Dynamic y-limits
    if not df_stats.empty:
        df_stats['lower'] = df_stats['mean'] - df_stats['std']
        df_stats['upper'] = df_stats['mean'] + df_stats['std']
        ymin = max(0, df_stats['lower'].min() * 0.65)
        ymax = min(100, df_stats['upper'].max() * 1.02)
        plt.ylim(ymin, ymax)
    else:
        plt.ylim(0, 1)

    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(loc='best', fontsize=13)
    plt.tight_layout()

    save_path = f"sprod_gen_vs_disc_comparison.pdf"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Plot saved to {save_path}")