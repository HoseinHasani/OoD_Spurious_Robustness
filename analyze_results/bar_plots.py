import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Directories
input_dir = "pickles"
output_dir = "method_bar_plots"
os.makedirs(output_dir, exist_ok=True)

# Dataset setup
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

# OOD method categories
feature_methods = ['sprod3', 'knn', 'mds', 'rmds', 'she']
output_methods = ['msp', 'mls', 'gradnorm', 'ebo', 'vim']
all_methods = feature_methods + output_methods
method_colors = dict(zip(all_methods, sns.color_palette("tab20", len(all_methods))))

# Metric to visualize
metric = "AUROC"

# Start processing
for dataset in dataset_list:
    ood = near_ood_dataset[dataset_list.index(dataset)]
    corr_tag = f"r{hard_correlations[dataset]}"
    readable_name = display_names[dataset]
    records = []

    # Parse files
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

        with open(os.path.join(input_dir, fname), 'rb') as f:
            result = pickle.load(f)

        records.append({
            "dataset": ds,
            "backbone": backbone,
            "method": method,
            "seed": seed,
            metric: result.get(metric, None)
        })

    df = pd.DataFrame(records)
    if df.empty:
        print(f"⚠️ No data for {dataset} ({corr_tag})")
        continue

    # Compute mean & std
    stats = df.groupby(['backbone', 'method'])[metric].agg(['mean', 'std']).reset_index()

    for backbone in stats['backbone'].unique():
        sub = stats[stats['backbone'] == backbone]

        # Ensure all methods present
        method_rows = []
        for m in all_methods:
            row = sub[sub['method'] == m]
            if row.empty:
                method_rows.append({'method': m, 'mean': 0, 'std': 0})
            else:
                method_rows.append({'method': m, 'mean': row['mean'].values[0], 'std': row['std'].values[0]})
        df_plot = pd.DataFrame(method_rows)

        # Bar plot
        colors = [method_colors[m] for m in df_plot['method']]
        plt.figure(figsize=(5, 6))
        bars = plt.bar(
            df_plot['method'],
            df_plot['mean'],
            yerr=df_plot['std'],
            capsize=4,
            color=colors,
            edgecolor='black'
        )

        # Labeling
        plt.title(f"{metric} for {readable_name} - {backbone}", fontsize=14)
        plt.xlabel("OOD Detection Method", fontsize=12)
        plt.ylabel(f"{metric} (mean ± std)", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=13)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        # Color code feature/output
        ax = plt.gca()
        for label in ax.get_xticklabels():
            method = label.get_text()
            if method in feature_methods:
                label.set_color('red')
            else:
                label.set_color('blue')

        # Y limits
        vis = df_plot[df_plot['mean'] > 0]
        if not vis.empty:
            ymin = max(0, (vis['mean'] - vis['std']).min() * 0.9)
            ymax = min(100, (vis['mean'] + vis['std']).max() * 1.05)
        else:
            ymin, ymax = 0, 1
        plt.ylim(ymin, ymax)

        # Save plot
        plt.tight_layout()
        outname = f"{dataset}_{backbone}.png"
        plt.savefig(os.path.join(output_dir, outname), dpi=150)
        plt.close()
        print(f"✅ Saved plot: {outname}")
