import os
import re
from collections import defaultdict

table_dir = "backbone_tables"
metric1 = "AUROC"
metric2 = "FPR@95"

# table_dir = "backbone_aupr_tables"
# metric1 = "AUPR_IN"
# metric2 = "AUPR_OUT"


auroc_table = defaultdict(dict)
fpr_table = defaultdict(dict)

methods = [
     'MSP',
     'Energy',
     'MLS',
     'KLM',
     'GNorm',
     'ReAct',
     'VIM',
     'MDS',
     'RMDS',
     'KNN',
     'SHE',
     'SPROD']


backbones = ['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'DINOv2-S', 
             'ViT-S', 'Swin-B', 'DeiT-B', 'ConvNeXt-B', 'BiT-R50x1']


backbone_name_dict = {
    "BiT-R50x1": "BiT",
    "ConvNeXt-B": "CvNXt",
    "DeiT-B": "DeiT",
    "Swin-B": "Swin",
    "ViT-S": "ViT",
    "DINOv2-S": "DINOv2",
    "ResNet-101": "R101",
    "ResNet-18": "R18",
    "ResNet-34": "R34",
    "ResNet-50": "R50"
}

backbones_names = [backbone_name_dict[b] for b in backbones]

method_pattern = re.compile(r"^(\w+|\w+-\w+|\w+\+?\w*)\s*&(?:.*)&\s*\$([\d.]+)")

for filename in os.listdir(table_dir):
    if filename.endswith(".tex"):
        backbone_name = filename.replace(".tex", "")
        filepath = os.path.join(table_dir, filename)
        with open(filepath, "r") as f:
            content = f.read()

        parts = content.split(r'\end{tabular}')
        if len(parts) < 2:
            continue

        auroc_rows = parts[0].splitlines()
        for line in auroc_rows:
            match = method_pattern.match(line.strip())
            if match:
                method, avg = match.groups()
                auroc_table[method][backbone_name] = float(avg)

        # Second table = FPR
        fpr_rows = parts[1].splitlines()
        for line in fpr_rows:
            match = method_pattern.match(line.strip())
            if match:
                method, avg = match.groups()
                fpr_table[method][backbone_name] = float(avg)

# backbones = sorted({b for d in auroc_table.values() for b in d})
# methods = sorted(set(auroc_table) | set(fpr_table))

def make_latex_table(data_dict, metric_name):
    table = "\\begin{tabular}{l" + "c" * (len(backbones) + 1) + "}\n"
    table += "\\toprule\n"
    table += "Method & " + " & ".join(backbones_names) + " & Avg. \\\\\n\\midrule\n"
    
    for k, method in enumerate(methods):
        if k == len(methods) - 1:
            table += r"\bottomrule" + "\n"
        row = [method]
        values = []
        for b in backbones:
            val = data_dict.get(method, {}).get(b, "---")
            if isinstance(val, float):
                values.append(val)
                row.append(f"{val:.1f}")
            else:
                row.append(val)
        # Compute average over available values
        avg = sum(values) / len(values) if values else "---"
        row.append(f"{avg:.1f}" if isinstance(avg, float) else avg)
        table += " & ".join(row) + " \\\\\n"
    
    table += "\\bottomrule\n\\end{tabular}"
    return f"\\begin{{table}}[t]\n\\small\n\\centering\n\\caption{{{metric_name} across backbones}}\n" + table + "\n\\end{table}"

# Generate tables
auroc_latex = make_latex_table(auroc_table, metric1)
fpr_latex = make_latex_table(fpr_table, metric2)

# Save or print result
print(auroc_latex)
print("\n\n" + fpr_latex)
