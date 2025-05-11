import os
from collections import defaultdict

directory = "pickles"

components = defaultdict(set)

for filename in os.listdir(directory):
    if '^' in filename and filename.endswith(".pkl"):
        parts = filename[:-4].split('^')  # Remove .pkl and split
        if len(parts) < 7:
            print(f"Warning: Unexpected format in file: {filename}")
            continue
        dataset, backbone, method, ood_set, correlation, seed = parts[:6]
        flag = "_".join(parts[6:])
        if not (seed.startswith("s") and seed[1:].isdigit()):
            print(f"Warning: Seed format unexpected in file: {filename}")
        components["dataset"].add(dataset)
        components["backbone"].add(backbone)
        components["method"].add(method)
        components["ood_set"].add(ood_set)
        components["correlation"].add(correlation)
        components["seed"].add(seed)
        components["flag"].add(flag)

for key in ["dataset", "backbone", "method", "ood_set", "correlation", "seed", "flag"]:
    print(f"{key}: {sorted(components[key])}")
