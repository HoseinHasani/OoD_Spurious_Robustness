import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Path containing the .npy files
path = 'time'

# Dictionary to store values by method
method_values = defaultdict(list)

# Read and parse the files
for fname in os.listdir(path):
    if fname.endswith('.npy'):
        parts = fname.split('_')
        method = parts[0]  # e.g., 'ebo' from 'ebo_resnet_50_20.npy'
        fpath = os.path.join(path, fname)
        value = np.load(fpath).item()  # assuming each .npy contains a single scalar
        method_values[method].append(value)

# Compute mean and std
methods = sorted(method_values.keys())
means = [np.mean(method_values[m]) for m in methods]
stds = [np.std(method_values[m]) for m in methods]

# Plot
plt.figure(figsize=(12, 6))
bars = plt.bar(methods, means, yerr=stds, capsize=5, color='skyblue')
plt.ylabel('Value')
plt.title('Comparison of Methods (mean ± std over seeds)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
