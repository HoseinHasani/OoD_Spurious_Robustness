import pandas as pd
from itertools import product

def generate_latex_table(csv_path, fixed_attrs, row_attr1='method', row_attr2=None,
                         col_attr_1order=None, col_attr_2order=None, col_attr_3order=None,
                         output_file="table.txt"):
    # Load CSV
    data = pd.read_csv(csv_path)
    
    # Validate configuration
    all_attrs = {row_attr1, row_attr2, col_attr_1order, col_attr_2order, col_attr_3order}
    all_attrs.discard(None)
    for attr in fixed_attrs:
        if attr in all_attrs:
            raise ValueError(f"Conflict: Attribute '{attr}' is fixed and also used in rows or columns.")
    
    # Apply fixed attributes
    for attr, value in fixed_attrs.items():
        data = data[data[attr] == value]
    
    # Get unique values for rows and columns
    row_values = data[row_attr1].unique()
    if row_attr2:
        row_groups = {val: data[data[row_attr1] == val][row_attr2].unique() for val in row_values}
    else:
        row_groups = None
    
    col_attrs = [col_attr_1order, col_attr_2order, col_attr_3order]
    col_attrs = [attr for attr in col_attrs if attr]  # Remove None
    col_values = [data[attr].unique() for attr in col_attrs]
    
    # Generate table structure
    table_lines = []
    table_lines.append("\\begin{table}[!htb]")
    table_lines.append("\\centering")
    table_lines.append("\\fontsize{13}{15}\\selectfont")
    table_lines.append("\\resizebox{0.9\\textwidth}{!}{%")
    table_lines.append("\\begin{tabular}{" + "c" * (1 + len(col_values)) + "}")
    table_lines.append("\\specialrule{1.5pt}{1pt}{1pt}")
    
    # Build column hierarchy
    header_lines = []
    for i, attr in enumerate(col_attrs):
        cols = [f"\\textbf{{{val}}}" for val in col_values[i]]
        cmid_rules = f"\\cmidrule(lr){{{2 + i * len(cols)}-{2 + (i + 1) * len(cols) - 1}}}"
        header_lines.append(" & " + " & ".join(cols) + " \\\\ " + cmid_rules)
    table_lines.extend(header_lines)
    
    # Add rows
    for row_val in row_values:
        if row_groups:
            group_values = row_groups.get(row_val, [])
        else:
            group_values = [row_val]
        
        for group_val in group_values:
            row_label = f"\\textbf{{{row_val}}}" if group_val == row_val else group_val
            row_data = []
            
            for col_comb in product(*col_values):
                query = data[(data[row_attr1] == row_val)]
                if row_groups:
                    query = query[query[row_attr2] == group_val]
                for col_attr, col_val in zip(col_attrs, col_comb):
                    query = query[query[col_attr] == col_val]
                
                if not query.empty:
                    mean_val = query['mean_val'].values[0]
                    std_val = query['std_val'].values[0] if 'std_val' in query else None
                    if pd.isna(std_val):
                        row_data.append(f"${mean_val}$")
                    else:
                        row_data.append(f"${mean_val}_{{\\textcolor{{gray}}{{\\pm {std_val}}}}}$")
                else:
                    print(f"Missing combination: {row_val}, {group_val}, {col_comb}")
                    row_data.append("-")
            
            table_lines.append(row_label + " & " + " & ".join(row_data) + " \\\\")
        if row_groups:
            table_lines.append("\\specialrule{1pt}{1pt}{1pt}")
    
    # Finish table
    table_lines.append("\\specialrule{1.5pt}{1pt}{1pt}")
    table_lines.append("\\end{tabular}}")
    table_lines.append("\\vspace{-0.1mm}")
    table_lines.append("\\caption{Your Table Caption}")
    table_lines.append("\\vspace{-1mm}")
    table_lines.append("\\label{tab:your_label}")
    table_lines.append("\\end{table}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(table_lines))
    print(f"LaTeX table written to {output_file}")
