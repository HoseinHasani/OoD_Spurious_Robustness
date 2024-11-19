import pandas as pd

# Example usage
csv_file = "data.csv"  # Path to your CSV file
row_attr = "method"  # Row attribute
col_attr = "metric"  # Column attribute
fixed_attrs = {
    "train_dataset": "CIFAR10",  # Fixed attribute example
}
output_file = "table_output.txt"

# Load the CSV file
df = pd.read_csv(csv_file)

# Filter the dataframe based on fixed attributes
for attr, value in fixed_attrs.items():
    df = df[df[attr] == value]

# Resolve duplicates by aggregating the data (e.g., taking the mean of duplicates)
df = df.groupby([row_attr, col_attr], as_index=False).agg(
    mean_val=('mean_val', 'mean'),  # Aggregate mean_val
    std_val=('std_val', 'mean')    # Aggregate std_val
)

# Create a pivot table with row_attr as rows and col_attr as columns
pivot_table = df.pivot(index=row_attr, columns=col_attr, values=['mean_val', 'std_val'])

# Start building the LaTeX table
latex_table = []
latex_table.append(r"\begin{table}[!htb]")
latex_table.append(r"\centering")
latex_table.append(r"\fontsize{13}{15}\selectfont")
latex_table.append(r"\resizebox{0.9\textwidth}{!}{%")
latex_table.append(r"\begin{tabular}{" + "c" * (len(pivot_table.columns.levels[1]) + 1) + "}")
latex_table.append(r"\specialrule{1.5pt}{1pt}{1pt}")

# Add column headers
col_headers = [col for col in pivot_table.columns.levels[1]]
latex_table.append(
    " & ".join([row_attr] + [f"\\textbf{{{col}}}" for col in col_headers]) + r" \\"
)
latex_table.append(r"\specialrule{1.5pt}{1pt}{1pt}")

# Add rows
for row in pivot_table.index:
    row_data = []
    for col in col_headers:
        mean_val = pivot_table.loc[row, ('mean_val', col)]
        std_val = pivot_table.loc[row, ('std_val', col)]
        
        if pd.isna(mean_val):
            row_data.append(" ")
        elif pd.isna(std_val):
            row_data.append(f"${mean_val:.2f}$")
        else:
            row_data.append(f"${mean_val:.2f}_{{\\textcolor{{gray}}{{\\pm {std_val:.2f}}}}}$")
    
    latex_table.append(f"{row} & " + " & ".join(row_data) + r" \\")

latex_table.append(r"\specialrule{1.5pt}{1pt}{1pt}")
latex_table.append(r"\end{tabular}}")
latex_table.append(r"\vspace{-0.1mm}")
latex_table.append(r"\caption{Generated table}")
latex_table.append(r"\vspace{-1mm}")
latex_table.append(r"\label{tab:generated}")
latex_table.append(r"\end{table}")

# Write LaTeX table to the output file
with open(output_file, "w") as f:
    f.write("\n".join(latex_table))
print(f"LaTeX table written to {output_file}")
