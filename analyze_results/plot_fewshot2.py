import os
import re
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_metric_across_shots(path, metric="AUROC", full_dataset_value=99.01):
    # Mapping for professional names
    variant_map = {
        "default": "Euclidean Distance",
        "cos": "Cosine Distance",
        "softmax": "Softmax - Euclidean5",
        "softmax_cos": "Softmax - Cosine"
    }

    corr_map = {
        "r50": "r=50%",
        "r90": "r=90%"
    }

    # Pattern to extract components from filenames
    pattern = re.compile(r"waterbirds_clip_ViT_B_16_sprod3_placesbg_(r\d+)_s(\d+)_([0-9]+)shot_([a-z_]+)\.pkl")

    # Collect data
    records = []

    for fname in os.listdir(path):
        match = pattern.match(fname)
        if match:
            corr, seed, shot, variant = match.groups()
            seed = int(seed)
            shot = int(shot)
            with open(os.path.join(path, fname), "rb") as f:
                metrics = pickle.load(f)
                value = metrics.get(metric, None)
                if value is not None:
                    records.append({
                        "correlation": corr_map.get(corr, corr),
                        "seed": seed,
                        "shot": shot,
                        "variant": variant_map.get(variant, variant),
                        metric: value
                    })

    if not records:
        print(f"No records found for metric '{metric}' in directory: {path}")
        return

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Compute mean and confidence interval
    def ci95(x):
        return 1.96 * np.std(x) / np.sqrt(len(x))

    df_stats = df.groupby(["correlation", "shot", "variant"])[metric].agg(
        ["mean", ci95]).reset_index()

    # Plot
    sns.set(style="whitegrid", font_scale=1.4)
    plt.figure(figsize=(8.2, 5.8))

    variants = df_stats["variant"].unique()
    palette = sns.color_palette("tab10", len(variants))

    for i, variant in enumerate(variants):
        if 'Softmax' not in variant:
            continue
        
        for corr in ["r=50%", "r=90%"]:
            subset = df_stats[(df_stats["variant"] == variant) & (df_stats["correlation"] == corr)]
            linestyle = "--" if corr == "r=50%" else "-"
            label = f"{variant} ({corr})"
            # if 'Cosine' in variant:
            #     label = f'Softmax - Cosine ({corr})'
            # else:
            #     label = f'Softmax - Cosine ({corr})'                
            plt.errorbar(
                subset["shot"],
                subset["mean"],
                yerr=subset["ci95"],
                label=label,
                linestyle=linestyle,
                linewidth=2.2,
                color=palette[i],
                marker="o",
                capsize=4
            )

    # Add reference line for full dataset
    plt.axhline(full_dataset_value, linestyle=":", color="black", linewidth=2.4, alpha=0.7)
    plt.text(
        df["shot"].max() + 0.5,
        full_dataset_value + 0.03,
        "(Full Dataset)",
        color="black",
        fontsize=13,
        va="center"
    )

    plt.xlabel("Number of Shots per Minority Group", fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.title(f"Low-Shot OOD Detection with Softmax Scores", fontsize=16)
    # plt.legend(title="Distance Variant (Correlation)", fontsize=11, title_fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.xticks(sorted(df["shot"].unique()))
    plt.tight_layout()
    # plt.show()

    save_dir = f"CLIP_plots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"CLIP_lowshot_{metric}_softmax.pdf")
    
    plt.savefig(save_path)
    
# Example usage
plot_metric_across_shots(path="pickles", metric="AUROC")
