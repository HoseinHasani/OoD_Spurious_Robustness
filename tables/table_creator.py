import pandas as pd

# Example usage
csv_file = "data.csv"  # Path to your CSV file
row_attr = "method"  # Row attribute
col_attr_1st = "correlation"  # Primary column attribute
col_attr_2nd = "backbone"  # Secondary column attribute (set to None for flat table)
col_attr_3rd = None#"metric"  # Tertiary column attribute (can be None)


fixed_attrs = {
    "train_dataset": "CIFAR10",  # Fixed attribute example
}
output_file = "table_output.txt"

# Load the CSV file
df = pd.read_csv(csv_file)

# Filter the dataframe based on fixed attributes
for attr, value in fixed_attrs.items():
    df = df[df[attr] == value]

# Aggregate duplicates and compute mean and standard deviation
grouping_columns = [row_attr, col_attr_1st]
if col_attr_2nd:
    grouping_columns.append(col_attr_2nd)
if col_attr_3rd:
    grouping_columns.append(col_attr_3rd)

df = df.groupby(
    grouping_columns,
    as_index=False
).agg(
    mean_val=('mean_val', 'mean'),
    std_val=('std_val', 'mean')
)
      
# Determine column spans
if col_attr_3rd:
    pivot_table = df.pivot(index=row_attr, columns=[col_attr_1st, col_attr_2nd, col_attr_3rd], values=['mean_val', 'std_val'])
elif col_attr_2nd:
    pivot_table = df.pivot(index=row_attr, columns=[col_attr_1st, col_attr_2nd], values=['mean_val', 'std_val'])
else:
    pivot_table = df.pivot(index=row_attr, columns=col_attr_1st, values=['mean_val', 'std_val'])
    

# Start building the LaTeX table
latex_table = []
latex_table.append(r"\begin{table}[!htb]")
latex_table.append(r"\centering")
latex_table.append(r"\fontsize{13}{15}\selectfont")
latex_table.append(r"\resizebox{0.9\textwidth}{!}{%")


# Determine column spans for multi-level headers
if col_attr_3rd:
    col_1st_headers = pivot_table.columns.levels[1]  # First level: metric
    col_2nd_headers = pivot_table.columns.levels[2]  # Second level: backbone
    col_3rd_headers = pivot_table.columns.levels[3]  # Third level: additional attribute
elif col_attr_2nd:
    col_1st_headers = pivot_table.columns.levels[1]  # First level: metric
    col_2nd_headers = pivot_table.columns.levels[2]  # Second level: backbone
else:
    col_1st_headers = pivot_table.columns.levels[1]  # First level: metric

# Define the number of columns in the LaTeX tabular
num_cols = len(pivot_table.columns) + 1  # Include the row attribute
latex_table.append(r"\begin{tabular}{" + "c" * num_cols + "}")
latex_table.append(r"\specialrule{1.5pt}{1pt}{1pt}")

# Create column headers
header_line = [r"\multirow{3}{*}{\textbf{" + row_attr + r"}}"]


if col_attr_3rd:
    # Top-level headers
    for col_1st in col_1st_headers:
        span = sum(
            1 for col_2nd in col_2nd_headers for col_3rd in col_3rd_headers
            if ('mean_val', col_1st, col_2nd, col_3rd) in pivot_table.columns or
               ('std_val', col_1st, col_2nd, col_3rd) in pivot_table.columns
        )
        header_line.append(rf"\multicolumn{{{span}}}{{c}}{{\textbf{{{col_1st}}}}}")
    latex_table.append(" & ".join(header_line) + r" \\")
    # Add cmidrule for first level
    col_start = 2
    for col_1st in col_1st_headers:
        span = sum(
            1 for col_2nd in col_2nd_headers for col_3rd in col_3rd_headers
            if ('mean_val', col_1st, col_2nd, col_3rd) in pivot_table.columns or
               ('std_val', col_1st, col_2nd, col_3rd) in pivot_table.columns
        )
        latex_table.append(rf"\cmidrule(lr){{{col_start}-{col_start + span - 1}}}")
        col_start += span

    # Second-level headers
    sub_header_line = [" "]
    for col_1st in col_1st_headers:
        for col_2nd in col_2nd_headers:
            span = sum(
                1 for col_3rd in col_3rd_headers
                if ('mean_val', col_1st, col_2nd, col_3rd) in pivot_table.columns or
                   ('std_val', col_1st, col_2nd, col_3rd) in pivot_table.columns
            )
            if span > 0:
                sub_header_line.append(rf"\multicolumn{{{span}}}{{c}}{{\textbf{{{col_2nd}}}}}")
    latex_table.append(" & ".join(sub_header_line) + r" \\")
    # Add cmidrule for second level
    col_start = 2
    for col_1st in col_1st_headers:
        for col_2nd in col_2nd_headers:
            span = sum(
                1 for col_3rd in col_3rd_headers
                if ('mean_val', col_1st, col_2nd, col_3rd) in pivot_table.columns or
                   ('std_val', col_1st, col_2nd, col_3rd) in pivot_table.columns
            )
            if span > 0:
                latex_table.append(rf"\cmidrule(lr){{{col_start}-{col_start + span - 1}}}")
                col_start += span

    # Third-level headers
    sub_sub_header_line = [" "]
    for col_1st in col_1st_headers:
        for col_2nd in col_2nd_headers:
            for col_3rd in col_3rd_headers:
                if ('mean_val', col_1st, col_2nd, col_3rd) in pivot_table.columns:
                    sub_sub_header_line.append(rf"\textbf{{{col_3rd}}}")
    latex_table.append(" & ".join(sub_sub_header_line) + r" \\")
    
elif col_attr_2nd:
    # Top-level headers
    header_line = [r"\multirow{2}{*}{\textbf{" + row_attr + r"}}"]
    for col_1st in col_1st_headers:
        num_sub_cols = sum(
            1 for col_2nd in col_2nd_headers
            if ('mean_val', col_1st, col_2nd) in pivot_table.columns or
               ('std_val', col_1st, col_2nd) in pivot_table.columns
        )
        header_line.append(rf"\multicolumn{{{num_sub_cols}}}{{c}}{{\textbf{{{col_1st}}}}}")
    latex_table.append(" & ".join(header_line) + r" \\")
    
    # Add cmidrule for first level
    col_start = 2
    for col_1st in col_1st_headers:
        num_sub_cols = sum(
            1 for col_2nd in col_2nd_headers
            if ('mean_val', col_1st, col_2nd) in pivot_table.columns or
               ('std_val', col_1st, col_2nd) in pivot_table.columns
        )
        if num_sub_cols > 0:
            latex_table.append(rf"\cmidrule(lr){{{col_start}-{col_start + num_sub_cols - 1}}}")
        col_start += num_sub_cols

    # Sub-level headers
    sub_header_line = [" "]
    for col_1st in col_1st_headers:
        for col_2nd in col_2nd_headers:
            if ('mean_val', col_1st, col_2nd) in pivot_table.columns:
                sub_header_line.append(rf"\textbf{{{col_2nd}}}")
    latex_table.append(" & ".join(sub_header_line) + r" \\")

else:
    # Flat headers
    header_line = [row_attr] + [f"\\textbf{{{col}}}" for col in pivot_table.columns.levels[1]]
    latex_table.append(" & ".join(header_line) + r" \\")
latex_table.append(r"\specialrule{1.5pt}{1pt}{1pt}")

# Add rows with data
for row in pivot_table.index:
    row_data = []
    if col_attr_3rd:
        for col_1st in col_1st_headers:
            for col_2nd in col_2nd_headers:
                for col_3rd in col_3rd_headers:
                    if ('mean_val', col_1st, col_2nd, col_3rd) in pivot_table.columns:
                        mean_val = pivot_table.loc[row, ('mean_val', col_1st, col_2nd, col_3rd)]
                        std_val = pivot_table.loc[row, ('std_val', col_1st, col_2nd, col_3rd)]

                        if pd.isna(mean_val):
                            row_data.append(" ")
                        elif pd.isna(std_val):
                            row_data.append(f"${mean_val:.2f}$")
                        else:
                            row_data.append(f"${mean_val:.2f}_{{\\textcolor{{gray}}{{\\pm {std_val:.2f}}}}}$")
                    else:
                        row_data.append(" ")
                        
    elif col_attr_2nd:
        for col_1st in col_1st_headers:
            for col_2nd in col_2nd_headers:
                if ('mean_val', col_1st, col_2nd) in pivot_table.columns:
                    mean_val = pivot_table.get(('mean_val', col_1st, col_2nd), {}).get(row, float('nan'))
                    std_val = pivot_table.get(('std_val', col_1st, col_2nd), {}).get(row, float('nan'))
                    
                    if pd.isna(mean_val):
                        row_data.append(" ")
                    elif pd.isna(std_val):
                        row_data.append(f"${mean_val:.2f}$")
                    else:
                        row_data.append(f"${mean_val:.2f}_{{\\textcolor{{gray}}{{\\pm {std_val:.2f}}}}}$")
                else:
                    row_data.append(" ")
                    
    else:
        for col in col_1st_headers:
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
