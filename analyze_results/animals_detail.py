import pandas as pd

# Load dataset
df = pd.read_csv("dataset_metadata.csv")

# Create a pivot table: rows are classes, columns are attributes, values are counts
pivot_table = df.pivot_table(index='class', columns='attribute', aggfunc='size', fill_value=0)

# Add total per class (row-wise sum)
pivot_table['Total'] = pivot_table.sum(axis=1)

# Add total per attribute (column-wise sum)
totals_row = pivot_table.sum(axis=0)
totals_row.name = 'Total'

# Append total row to the pivot table
pivot_table = pd.concat([pivot_table, pd.DataFrame([totals_row])])

# Convert the table to LaTeX
latex_table = pivot_table.to_latex(index=True, caption='Samples per Class and Attribute', label='tab:class_attribute_counts')

# Save LaTeX table to a .tex file (optional)
with open("class_attribute_table.tex", "w") as f:
    f.write(latex_table)

# Print the LaTeX table
print(latex_table)
