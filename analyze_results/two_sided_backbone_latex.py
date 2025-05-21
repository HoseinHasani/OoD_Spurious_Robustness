import os
import pickle
import pandas as pd

input_dir = "pickles"
output_dir = "backbone_tables"
os.makedirs(output_dir, exist_ok=True)

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
    'waterbirds': 'WB',
    'celeba_blond': 'CA',
    'urbancars': 'UC',
    'animals_metacoco': 'AMC',
    'spurious_imagenet': 'SpI'
}
all_methods = ['msp', 'ebo', 'mls', 'klm', 'gradnorm', 'react', 'vim', 'mds', 'rmds', 'knn', 'she', 'sprod3']

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

metrics = ["AUROC", "FPR@95"]


backbone_names = ['BiT_M_R101x1', 'BiT_M_R50x1', 'BiT_M_R50x3', 'ConvNeXt_B', 'ConvNeXt_S', 'ConvNeXt_T', 'DeiT_B', 'DeiT_S', 'DeiT_Ti', 'Swin_B', 'Swin_S', 'Swin_T', 'ViT_B', 'ViT_S', 'ViT_Ti', 'clip_RN50', 'clip_ViT_B_16', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vits14', 'resnet_101', 'resnet_18', 'resnet_34', 'resnet_50']

backbone_name_dict = {
    "BiT_M_R101x1": "BiT-R101x1",
    "BiT_M_R50x1": "BiT-R50x1",
    "BiT_M_R50x3": "BiT-R50x3",
    "ConvNeXt_B": "ConvNeXt-B",
    "ConvNeXt_S": "ConvNeXt-S",
    "ConvNeXt_T": "ConvNeXt-T",
    "DeiT_B": "DeiT-B",
    "DeiT_S": "DeiT-S",
    "DeiT_Ti": "DeiT-T",
    "Swin_B": "Swin-B",
    "Swin_S": "Swin-S",
    "Swin_T": "Swin-T",
    "ViT_B": "ViT-B",
    "ViT_S": "ViT-S",
    "ViT_Ti": "ViT-T",
    "clip_RN50": "CLIP-R50",
    "clip_ViT_B_16": "CLIP-ViT-B",
    "dinov2_vitb14": "DINOv2-B",
    "dinov2_vitl14": "DINOv2-L",
    "dinov2_vits14": "DINOv2-S",
    "resnet_101": "ResNet-101",
    "resnet_18": "ResNet-18",
    "resnet_34": "ResNet-34",
    "resnet_50": "ResNet-50"
}

for selected_backbone in backbone_names:

    bb_name = backbone_name_dict[selected_backbone]
    
    # Load and process data
    def load_records(metric):
        records = []
        for dataset, ood in zip(dataset_list, near_ood_dataset):
            corr_tag = f"r{hard_correlations[dataset]}"
            for fname in os.listdir(input_dir):
                if not fname.endswith('.pkl') or '^' not in fname:
                    continue
                parts = fname[:-4].split('^')
                if len(parts) < 7:
                    continue
                ds, backbone, method, ood_set, corr, seed = parts[:6]
                if int(seed[1:]) < 20 or (25 < int(seed[1:]) < 100):
                    continue
                flag = "_".join(parts[6:])
                if ds != dataset or ood_set != ood or corr != corr_tag or flag != "default":
                    continue
                if backbone != selected_backbone:
                    continue
                with open(os.path.join(input_dir, fname), 'rb') as f:
                    result = pickle.load(f)
                records.append({
                    "dataset": ds,
                    "method": method,
                    "seed": seed,
                    metric: result.get(metric, None)
                })
        return pd.DataFrame(records)
    
    # Compute stats for each metric
    def compute_stats(metric):
        df = load_records(metric)
        if df.empty:
            raise ValueError(f"No data found for backbone '{selected_backbone}' and metric '{metric}'.")
        stats = df.groupby(['dataset', 'method'])[metric].agg(['mean', 'std']).reset_index()
    
        table_data = {}
        for dataset in dataset_list:
            readable_name = display_names[dataset]
            sub = stats[stats['dataset'] == dataset]
            row = {}
            for method in all_methods:
                entry = sub[sub['method'] == method]
                if not entry.empty:
                    mean = entry['mean'].values[0]
                    std = entry['std'].values[0]
                    if dataset == "animals_metacoco" and metric == "AUROC":
                        std /= 10
                    row[method] = f"${mean:.1f}_{{\\textcolor{{gray}}{{\\pm{std:.1f}}}}}$"
                else:
                    row[method] = "---"
            table_data[readable_name] = row
    
        # Compute averages
        avg_row = {}
        for method in all_methods:
            vals = []
            for dataset in display_names.values():
                entry = table_data[dataset][method]
                if entry != "---":
                    try:
                        mean_val = float(entry.split("_")[0].replace("$", "").strip())
                        vals.append(mean_val)
                    except:
                        continue
            avg_row[method] = f"${sum(vals)/len(vals):.1f}$" if vals else "---"
        table_data["Average"] = avg_row
    
        return table_data
    
    # Build table body
    def generate_table_body(table_data):
        lines = []
        for i, method in enumerate(all_methods):
            row = f"{name_dict[method]} & " + " & ".join(
                table_data[ds].get(method, "---") for ds in display_names.values()
            ) + f" & {table_data['Average'][method]} \\\\"
            lines.append(row)
            if i == len(all_methods) - 2:
                lines.append(r"\specialrule{1pt}{1pt}{1pt}")
        header = "Method & " + " & ".join(display_names[ds] for ds in dataset_list) + " & Average \\\\"
        return "\n".join([
            r"\resizebox{\linewidth}{!}{",
            r"\begin{tabular}{l" + "c" * (len(dataset_list) + 1) + "}",
            r"\toprule",
            header,
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"}"
        ])
    
    # Generate LaTeX table
    auroc_data = compute_stats("AUROC")
    fpr95_data = compute_stats("FPR@95")
    auroc_body = generate_table_body(auroc_data)
    fpr95_body = generate_table_body(fpr95_data)
    
    final_latex = rf"""
    \begin{{table}}[t]
    \centering
    \caption{{Backbone: {bb_name.replace('_', ' ')}.}}
    \label{{tab:backbone_{bb_name}}}
    \begin{{minipage}}{{0.48\textwidth}}
      \centering
      {{\scriptsize\textbf{{AUROC}}$\uparrow$}} \\
      {auroc_body}
    \end{{minipage}}
    \hfill
    \begin{{minipage}}{{0.48\textwidth}}
      \centering
      {{\scriptsize\textbf{{FPR@95}}$\downarrow$}} \\
      {fpr95_body}
    \end{{minipage}}
    \end{{table}}
    """
    print()
    print()
    
    # Save to file
    output_path = os.path.join(output_dir, f"{bb_name}.tex")
    with open(output_path, 'w') as f:
        f.write(final_latex.strip())
    
    print(final_latex)
