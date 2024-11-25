import os
import pickle
import pandas as pd
import numpy as np

# Define the path where the pickle files are stored
results_dir = "results/softmax_experiment/"

# List all methods (assuming the filenames follow the pattern 'animals_<method_name>_results.pkl')
methods = ['mrefinedprototypical']
model_name = 'resnet_50'

# Initialize a dictionary to store summary results
summary_results = {}

# Loop through each method
for method in methods:
    method_results = {}
    
    # Construct the path to the pickle file for this method
    file_path = os.path.join(results_dir, f"animals_{method}_results.pkl")
    
    # Load the pickle file for the method
    with open(file_path, "rb") as f:
        results_dict = pickle.load(f)
    
    # Initialize lists to store the relevant rows for 'nearood' and 'farood' classes
    nearood_metrics = []
    farood_metrics = []
    
    # Variables to keep track of best and worst AUROC for 'nearood'
    best_auroc = -np.inf
    worst_auroc = np.inf
    best_class = None
    worst_class = None
    best_class_metrics = None
    worst_class_metrics = None
    
    # Loop through each key (which corresponds to a specific OOD class) in the results_dict
    for key in results_dict:
        if model_name in key:  # Check if the model name is in the key (e.g., 'resnet_50' or 'resnet_18')
            # Extract the OOD class name from the key (the last part after '_')
            class_name = key.split('_')[-1]  # e.g., 'cat', 'dog', etc.
            
            # Extract the DataFrame for the current class
            metrics_df = results_dict[key]
            
            # Ensure the DataFrame is in a format we can calculate mean and std on (we need to convert from strings)
            metrics_df = metrics_df.replace(r' ±.*', '', regex=True).astype(float)
            
            # Now we need to filter for nearood and farood rows within the current class' DataFrame
            if 'nearood' in metrics_df.index:
                nearood_row = metrics_df.loc['nearood']
                nearood_metrics.append(nearood_row)
                
                # Track the best and worst class based on AUROC for nearood
                auroc = nearood_row['AUROC']
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_class = class_name
                    best_class_metrics = nearood_row
                if auroc < worst_auroc:
                    worst_auroc = auroc
                    worst_class = class_name
                    worst_class_metrics = nearood_row

            if 'farood' in metrics_df.index:
                farood_row = metrics_df.loc['farood']
                farood_metrics.append(farood_row)

    # Calculate the mean and std for 'nearood' and 'farood' (across all OOD classes)
    if nearood_metrics:
        nearood_metrics_array = np.array(nearood_metrics)
        nearood_mean = np.mean(nearood_metrics_array, axis=0)
        nearood_std = np.std(nearood_metrics_array, axis=0)
        method_results['nearood_mean_metrics'] = np.round(nearood_mean, 2)
        method_results['nearood_std_metrics'] = np.round(nearood_std, 2)
    
    if farood_metrics:
        farood_metrics_array = np.array(farood_metrics)
        farood_mean = np.mean(farood_metrics_array, axis=0)
        farood_std = np.std(farood_metrics_array, axis=0)
        method_results['farood_mean_metrics'] = np.round(farood_mean, 2)
        method_results['farood_std_metrics'] = np.round(farood_std, 2)
    
    # Store the best and worst class based on AUROC for 'nearood'
    method_results['best_class_nearood'] = best_class
    method_results['worst_class_nearood'] = worst_class
    
    # Store the metrics for best and worst class (rounded to 2 decimal places)
    method_results['best_class_metrics'] = np.round(best_class_metrics, 2)
    method_results['worst_class_metrics'] = np.round(worst_class_metrics, 2)
    
    # Store the results for this method
    summary_results[method] = method_results

# Save the summary results into a pickle file
output_file_path = f"results/softmax_experiment/animals_summary_results_{method}_{model_name}.pkl"
print(summary_results)
with open(output_file_path, "wb") as f:
    pickle.dump(summary_results, f)

print("Summary results with mean, std, best and worst class based on AUROC, along with their metrics, have been saved to:", output_file_path)