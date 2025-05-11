import os
import pickle
import pandas as pd
from collections import defaultdict

# Directory containing the result files
directory = "pickles"

# Initialize list for structured records
records = []

# Load each result and parse its fields
for filename in os.listdir(directory):
    if '^' in filename and filename.endswith(".pkl"):
        parts = filename[:-4].split('^')
        if len(parts) < 7:
            continue
        dataset, backbone, method, ood_set, correlation, seed = parts[:6]
        flag = "_".join(parts[6:])  # Join remaining parts for flag

        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            result = pickle.load(f)
        
        # Add one row per file with all metadata + metrics
        record = {
            "dataset": dataset,
            "backbone": backbone,
            "method": method,
            "ood_set": ood_set,
            "correlation": correlation,
            "seed": seed,
            "flag": flag,
        }
        record.update(result)  # Add metrics: FPR@95, AUROC, etc.
        records.append(record)

# Convert to DataFrame
df = pd.DataFrame.from_records(records)

# --- CONFIGURABLE COMPARISON PARAMETERS ---
# Example: fix backbone and correlation; compare method vs. ood_set; average over seeds
fixed_fields = ["backbone", "correlation"]
fields_to_compare = ["method", "ood_set"]
fields_for_average = ["seed"]

# Metric to compare
metric = "FPR@95"

# Filter rows matching the fixed field values
# Example: backbone=resnet_18, correlation=r0.9
filters = {
    "backbone": "resnet_18",
    "correlation": "r0.9",
    "flag": "default"
}
for key, value in filters.items():
    df = df[df[key] == value]

# Group and average
group_keys = fields_to_compare.copy()
df_avg_
