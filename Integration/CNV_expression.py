import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal

# Load the CNV matrix
cnv_matrix = pd.read_csv('cnv_matrix.tsv', sep='\t', index_col=0)
cnv_matrix = cnv_matrix * 2 - 2

# Load the raw count matrix
raw_count_matrix = pd.read_csv('raw_count_matrix.txt', sep='\t', index_col=0)

# Debugging: Print dimensions of the dataframes
print(f"CNV matrix dimensions: {cnv_matrix.shape}")
print(f"Raw count matrix dimensions (before filtering): {raw_count_matrix.shape}")

# Step 0: Filter the raw count matrix
# Filter genes: Keep genes expressed in at least 200 cells
gene_filter = (raw_count_matrix > 0).sum(axis=1) >= 200  # Genes expressed in >= 200 cells
filtered_raw_count_matrix = raw_count_matrix[gene_filter]

# Filter cells: Keep cells expressing at least 3 genes
cell_filter = (filtered_raw_count_matrix > 0).sum(axis=0) >= 3  # Cells expressing >= 3 genes
filtered_raw_count_matrix = filtered_raw_count_matrix.loc[:, cell_filter]

# Debugging: Print dimensions of the filtered raw count matrix
print(f"Filtered raw count matrix dimensions: {filtered_raw_count_matrix.shape}")

# Step 1: Z-normalize the filtered raw count matrix
# Z-score = (X - mean) / std
z_score_matrix = filtered_raw_count_matrix.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

# Debugging: Print dimensions of the z-score matrix
print(f"Z-score matrix dimensions: {z_score_matrix.shape}")

# Ensure the genes and cells match between the two matrices
assert all(cnv_matrix.index == z_score_matrix.index), "Gene names do not match!"
assert all(cnv_matrix.columns == z_score_matrix.columns), "Cell names do not match!"

# Step 2: Map CNV to Z-scores
# Create a DataFrame to store the results
results = []

for cell in cnv_matrix.columns:
    for gene in cnv_matrix.index:
        cnv = cnv_matrix.loc[gene, cell]
        z_score = z_score_matrix.loc[gene, cell]
        results.append({'Gene': gene, 'Cell': cell, 'CNV': cnv, 'Z-score': z_score})

results_df = pd.DataFrame(results)

# Debugging: Print the first few rows of the results dataframe
print("First few rows of the results dataframe:")
print(results_df.head())

# Step 3: Perform statistical testing (Kruskal-Wallis test)
# Group z-scores by CNV category
cnv_groups = [results_df[results_df['CNV'] == cnv]['Z-score'] for cnv in results_df['CNV'].unique()]
statistic, p_value = kruskal(*cnv_groups)

print(f"Kruskal-Wallis Test Results: Statistic = {statistic}, p-value = {p_value}")

# Step 4: Create a boxplot of z-scores per CNV category
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='CNV',
    y='Z-score',
    data=results_df,
    palette='viridis',  # Use Viridis color palette
    showfliers=False  # Hide outlier dots
)
plt.title('Gene Expression (z-score normalized) per Copy Number Variation')
plt.xlabel('Copy Number Variation (CNV)')
plt.ylabel('Z-score of Gene Expression')

# # Add statistical test results to the plot
# plt.text(
#     0.5,  # x position
#     2.5,  # y position
#     f'Kruskal-Wallis Test: p = {p_value:.4f}',  # Display p-value
#     fontsize=12,
#     ha='center'
# )

# Save as a file (e.g., PNG, PDF)
plt.savefig('zscore_cnv_boxplot.png', dpi=300, bbox_inches='tight')  # Save as PNG
# plt.savefig('zscore_cnv_boxplot.pdf', bbox_inches='tight')  # Save as PDF (optional)

plt.close()  # Close the plot to free memory

# Step 5: Create a violin plot of z-scores per CNV category
plt.figure(figsize=(10, 6))
ax = sns.violinplot(
    x='CNV',
    y='Z-score',
    data=results_df,
    palette='viridis',  # Use Viridis color palette
    inner='quartile',  # Show quartiles inside the violin plot
    gridsize=1000  # Increase the number of grid points for smoother violins
)

# Set y-axis limits
upp = results_df['Z-score'].max()  # Upper limit for y-axis
# Set y-axis limits
ax.set_ylim(-2, 3)

plt.title('Gene Expression (z-score normalized) per Copy Number Variation')
plt.xlabel('Copy Number Variation (CNV)')
plt.ylabel('Z-score of Gene Expression')

# Save as a file (e.g., PNG, PDF)
plt.savefig('zscore_cnv_violinplot.png', dpi=300, bbox_inches='tight')  # Save as PNG
# plt.savefig('zscore_cnv_violinplot.pdf', bbox_inches='tight')  # Save as PDF (optional)

plt.close()  # Close the plot to free memory




import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import loompy as lp
import pandas as pd
import numpy as np
import seaborn as sns

## CREATE REGULON DF
# Path to the .loom file
loom_file_path = "/work/project/ladcol_020/scGRNi/RNA/SCENIC/ccRCC_GBM/C3N-00495-T1_CPT0078510004/pyscenic_output.loom"

# Open the .loom file
lf = lp.connect(loom_file_path, mode='r', validate=False)

# Load the CNV matrix
cnv_matrix = pd.read_csv('cnv_matrix.tsv', sep='\t', index_col=0)
cnv_matrix = cnv_matrix * 2 - 2

# Load the raw count matrix
raw_count_matrix = pd.read_csv('raw_count_matrix.txt', sep='\t', index_col=0)

# Step 1: Filter cells where `cell_type.harmonized.cancer` is "Tumor"
tumor_cell_indices = np.where(lf.ca['cell_type.harmonized.cancer'] == "Tumor")[0]
tumor_cell_ids = lf.ca['CellID'][tumor_cell_indices]

# Step 2: Extract RegulonsAUC for tumor cells
# Convert the 1D array of tuples into a 2D array
regulons_auc = np.array(lf.ca['RegulonsAUC'].tolist())  # Shape: (6855, num_regulons)
regulons_auc_tumor = regulons_auc[tumor_cell_indices, :]  # Filter for tumor cells

# Step 3: Extract regulon names from the column names of RegulonsAUC
# Assuming the regulon names are stored in the column names of RegulonsAUC
regulon_names = lf.ca['RegulonsAUC'].dtype.names  # Extract column names (regulon names)

# Step 4: Create a DataFrame with regulons as rows and tumor cells as columns
regulons_auc_df = pd.DataFrame(regulons_auc_tumor.T, index=regulon_names, columns=tumor_cell_ids)

# Step 5: Add a "target_genes" column
# Convert the 1D array of tuples into a 2D binary matrix
regulons_binary_matrix = np.array(lf.ra['Regulons'].tolist())  # Shape: (4928, num_regulons)

# Create a dictionary to map regulons to their target genes
regulon_to_genes = {}
for i, regulon in enumerate(regulon_names):
    # Get the indices of genes that belong to this regulon (where value is 1)
    target_gene_indices = np.where(regulons_binary_matrix[:, i] == 1)[0]
    # Get the gene names for these indices
    target_genes = lf.ra['Gene'][target_gene_indices]
    # Add to the dictionary
    regulon_to_genes[regulon] = list(target_genes)

# Add the "target_genes" column to the DataFrame
regulons_auc_df['target_genes'] = regulons_auc_df.index.map(regulon_to_genes)

# Step 6: Close the .loom file
lf.close()

# Display the resulting DataFrame
print(regulons_auc_df.head())

# Compute the average activity across all columns except 'target_genes'
regulons_auc_df['average_activity'] = regulons_auc_df.iloc[:, :-1].mean(axis=1)

# Sort by 'average_activity' in descending order
sorted_df = regulons_auc_df.sort_values(by='average_activity', ascending=False)

# Print the top 10 highest
print(sorted_df.head(10))

# Count the number of target genes for each regulon in the sorted DataFrame
sorted_df['num_target_genes'] = sorted_df['target_genes'].apply(len)

# Print the top 10 with the number of target genes
print(sorted_df[['average_activity', 'num_target_genes']].head(10))

# Step 1: Calculate the average activity of each regulon (ignore the target_genes column)
regulons_avg_activity = regulons_auc_df.drop(['target_genes', 'average_activity'], axis=1).mean(axis=1)

# Step 2: Extract the target genes for all regulons
all_target_genes = regulons_auc_df['target_genes']

# Step 3: Calculate the average copy number and standard deviation of the target genes for all regulons
average_copy_numbers = []
copy_number_std_devs = []  # To store standard deviations

for target_genes in all_target_genes:
    # Ensure target_genes is a list of strings
    target_genes_list = [gene.strip() for gene in target_genes if gene.strip() in cnv_matrix.index]
    if target_genes_list:
        # Calculate the average copy number and standard deviation for the target genes
        average_copy_number = cnv_matrix.loc[target_genes_list].mean().mean()
        std_dev = cnv_matrix.loc[target_genes_list].mean().std()  # Standard deviation across genes
    else:
        # If no target genes are found in the CNV matrix, set the copy number and std dev to NaN
        average_copy_number = np.nan
        std_dev = np.nan
    average_copy_numbers.append(average_copy_number)
    copy_number_std_devs.append(std_dev)

# Step 4: Create a DataFrame for plotting
plot_data = pd.DataFrame({
    'Average Copy Number': average_copy_numbers,
    'Copy Number Std Dev': copy_number_std_devs,  # Add standard deviation
    'Regulon Average Activity': regulons_avg_activity
})

# Step 5: Plot the data with error bars
plt.figure(figsize=(10, 6))

# Create scatterplot with flipped axes
scatterplot = sns.scatterplot(
    x='Average Copy Number',  # Flipped to x-axis
    y='Regulon Average Activity',  # Flipped to y-axis
    hue='Regulon Average Activity',
    data=plot_data,
    s=40,
    palette='viridis',
    alpha=1,
    edgecolor='none',
    zorder=3
)

# Add error bars with flipped axes
plt.errorbar(
    x=plot_data['Average Copy Number'],  # Flipped to x-axis
    y=plot_data['Regulon Average Activity'],  # Flipped to y-axis
    xerr=plot_data['Copy Number Std Dev'],  # Error bars now on x-axis
    fmt='none',
    ecolor='gray',
    elinewidth=1,
    capsize=2,
    alpha=0.4,
    zorder=2
)

# Get the top 5 regulons with the highest activity
top_regulons = plot_data.nlargest(5, 'Regulon Average Activity')

# Add text annotations for the top 5 regulons
for regulon, row in top_regulons.iterrows():
    plt.text(
        row['Average Copy Number'],  # Flipped to x-axis
        row['Regulon Average Activity'] + 0.001,  # Flipped to y-axis
        regulon,
        fontsize=10,
        ha='center',
        va='bottom'
    )

# Customize the plot
plt.xlabel('Average Copy Number of Regulon Target Genes')  # Flipped to x-axis
plt.ylabel('Average Regulon Activity')  # Flipped to y-axis
plt.minorticks_on()
plt.grid(True, which="major", linestyle='--', alpha=0.7, linewidth=0.25)
plt.legend([],[], frameon=False)

# Adjust layout and display
plt.tight_layout()
plt.savefig('regulon_activity_cnv_scatterplot.png', dpi=300, bbox_inches='tight')
plt.close()


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal

# Load the CNV matrix
cnv_matrix = pd.read_csv('cnv_matrix.tsv', sep='\t', index_col=0)

# Load the raw count matrix
raw_count_matrix = pd.read_csv('raw_count_matrix.txt', sep='\t', index_col=0)

# Debugging: Print dimensions of the dataframes
print(f"CNV matrix dimensions: {cnv_matrix.shape}")
print(f"Raw count matrix dimensions (before filtering): {raw_count_matrix.shape}")

# Step 0: Filter the raw count matrix
# Filter genes: Keep genes expressed in at least 200 cells
gene_filter = (raw_count_matrix > 0).sum(axis=1) >= 200  # Genes expressed in >= 200 cells
filtered_raw_count_matrix = raw_count_matrix[gene_filter]

# Filter cells: Keep cells expressing at least 3 genes
cell_filter = (filtered_raw_count_matrix > 0).sum(axis=0) >= 3  # Cells expressing >= 3 genes
filtered_raw_count_matrix = filtered_raw_count_matrix.loc[:, cell_filter]

# Debugging: Print dimensions of the filtered raw count matrix
print(f"Filtered raw count matrix dimensions: {filtered_raw_count_matrix.shape}")

# Step 1: Z-normalize the filtered raw count matrix
# Z-score = (X - mean) / std
z_score_matrix = filtered_raw_count_matrix.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

# Debugging: Print dimensions of the z-score matrix
print(f"Z-score matrix dimensions: {z_score_matrix.shape}")

# Ensure the genes and cells match between the two matrices
assert all(cnv_matrix.index == z_score_matrix.index), "Gene names do not match!"
assert all(cnv_matrix.columns == z_score_matrix.columns), "Cell names do not match!"

# Step 2: Map CNV to Z-scores
# Create a DataFrame to store the results
results = []

for cell in cnv_matrix.columns:
    for gene in cnv_matrix.index:
        cnv = cnv_matrix.loc[gene, cell]
        z_score = z_score_matrix.loc[gene, cell]
        results.append({'Gene': gene, 'Cell': cell, 'CNV': cnv, 'Z-score': z_score})

results_df = pd.DataFrame(results)

# Debugging: Print the first few rows of the results dataframe
print("First few rows of the results dataframe:")
print(results_df.head())

# Step 3: Perform statistical testing (Kruskal-Wallis test)
# Group z-scores by CNV category
cnv_groups = [results_df[results_df['CNV'] == cnv]['Z-score'] for cnv in results_df['CNV'].unique()]
statistic, p_value = kruskal(*cnv_groups)

print(f"Kruskal-Wallis Test Results: Statistic = {statistic}, p-value = {p_value}")

# Step 4: Create a boxplot of z-scores per CNV category
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='CNV',
    y='Z-score',
    data=results_df,
    palette='viridis',  # Use Viridis color palette
    showfliers=False  # Hide outlier dots
)
plt.title('Z-scores of Gene Expression per Copy Number Variation')
plt.xlabel('Copy Number Variation (CNV)')
plt.ylabel('Z-score of Gene Expression')

# Add statistical test results to the plot
plt.text(
    0.5,  # x position
    2.5,  # y position
    f'Kruskal-Wallis Test: p = {p_value:.4f}',  # Display p-value
    fontsize=12,
    ha='center'
)

# Save as a file (e.g., PNG, PDF)
plt.savefig('zscore_cnv_boxplot.png', dpi=300, bbox_inches='tight')  # Save as PNG
# plt.savefig('zscore_cnv_boxplot.pdf', bbox_inches='tight')  # Save as PDF (optional)

plt.close()  # Close the plot to free memory

# Step 5: Create a violin plot of z-scores per CNV category
plt.figure(figsize=(10, 6))
ax = sns.violinplot(
    x='CNV',
    y='Z-score',
    data=results_df,
    palette='viridis',  # Use Viridis color palette
    inner='quartile',  # Show quartiles inside the violin plot
    gridsize=1000  # Increase the number of grid points for smoother violins
)

# Set y-axis limits
upp = results_df['Z-score'].max()  # Upper limit for y-axis
# Set y-axis limits
ax.set_ylim(-2, 3)

plt.title('Z-scores of Gene Expression per Copy Number Variation (Violin Plot)')
plt.xlabel('Copy Number Variation (CNV)')
plt.ylabel('Z-score of Gene Expression')

# Save as a file (e.g., PNG, PDF)
plt.savefig('zscore_cnv_violinplot.png', dpi=300, bbox_inches='tight')  # Save as PNG
# plt.savefig('zscore_cnv_violinplot.pdf', bbox_inches='tight')  # Save as PDF (optional)

plt.close()  # Close the plot to free memory

