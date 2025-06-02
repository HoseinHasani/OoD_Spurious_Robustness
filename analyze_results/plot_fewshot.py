import os
import re
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

use_softmax = False

def plot_metric_across_shots(path, metric="AUROC", full_dataset_value=2.94):
    # Mapping for professional names
    variant_map = {
        "default": "Euclidean Distance",
        "cos": "Cosine Distance",
        "softmax": "Softmax",
        "softmax_cos": "Softmax + Cosine"
    }

    corr_map = {
        "r50": "r=50%",
        "r90": "r=90%"
    }

    pattern = re.compile(r"waterbirds_clip_ViT_B_16_sprod3_placesbg_(r\d+)_s(\d+)_([0-9]+)shot_([a-z_]+)\.pkl")
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

    df = pd.DataFrame(records)

    def ci95(x):
        return 1.96 * np.std(x) / np.sqrt(len(x))

    df_stats = df.groupby(["correlation", "shot", "variant"])[metric].agg(["mean", ci95]).reset_index()

    sns.set(style="whitegrid", font_scale=1.4)
    plt.figure(figsize=(8, 6))

    variants = df_stats["variant"].unique()
    palette = sns.color_palette("tab10", len(variants))
    color_map = dict(zip(variants, palette))

    for variant in variants:
        
        if not use_softmax:
            
            if 'Softmax' in variant:
                continue
        else:
            if 'Softmax' not in variant:
                continue
            
        for corr in ["r=50%", "r=90%"]:
            linestyle = "--" if corr == "r=50%" else "-"
            subset = df_stats[(df_stats["variant"] == variant) & (df_stats["correlation"] == corr)]
            plt.errorbar(
                subset["shot"],
                subset["mean"],
                yerr=subset["ci95"],
                label=None,
                linestyle=linestyle,
                linewidth=2.2,
                color=color_map[variant],
                marker="o",
                capsize=4
            )

    # Add full dataset reference line
    plt.axhline(full_dataset_value, linestyle=":", color="black", linewidth=2.4, alpha=0.7)
    plt.text(
        df["shot"].max() + 0.5,
        full_dataset_value + 0.03,
        "(Full Dataset)",
        color="black",
        fontsize=13,
        va="center"
    )

    # Custom legend handles
    variant_handles = [
        Line2D([0], [0], color=color_map["Cosine Distance"], lw=2.2, label="Cosine Distance"),
        Line2D([0], [0], color=color_map["Euclidean Distance"], lw=2.2, label="Euclidean Distance")
    ]

    correlation_handles = [
        Line2D([0], [0], color="black", linestyle="-", lw=2, label="r=90%"),
        Line2D([0], [0], color="black", linestyle="--", lw=2, label="r=50%")
    ]

    plt.xlabel("Number of Shots per Minority Group", fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.title(f"Low-Shot OOD Detection with CLIP Embeddings", fontsize=16)
    first_legend = plt.legend(handles=variant_handles, loc="upper right", title="Distance Metric", fontsize=12, title_fontsize=12)
    plt.gca().add_artist(first_legend)
    plt.legend(handles=correlation_handles, loc="upper center", title="Spurious Correlation", fontsize=12, title_fontsize=12)

    plt.xticks(sorted(df["shot"].unique()))
    plt.tight_layout()
    # plt.show()
    
    save_dir = f"CLIP_plots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"CLIP_lowshot_{metric}_{use_softmax}.pdf")
    
    plt.savefig(save_path)
    
# Example usage
plot_metric_across_shots(path="pickles", metric="FPR@95")
