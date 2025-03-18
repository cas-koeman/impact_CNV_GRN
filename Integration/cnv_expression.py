import loompy as lp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
import os

class RegulonAnalysis:
    def __init__(self, loom_file_path, cnv_matrix_path, raw_count_matrix_path):
        self.loom_file_path = loom_file_path
        self.cnv_matrix_path = cnv_matrix_path
        self.raw_count_matrix_path = raw_count_matrix_path
        self.regulons_auc_df = None
        self.sorted_df = None
        self.results_df = None

    def load_and_process_loom(self):
        # Open the .loom file
        lf = lp.connect(self.loom_file_path, mode='r', validate=False)

        # Step 1: Filter cells where `cell_type.harmonized.cancer` is "Tumor"
        tumor_cell_indices = np.where(lf.ca['cell_type.harmonized.cancer'] == "Tumor")[0]
        tumor_cell_ids = lf.ca['CellID'][tumor_cell_indices]

        # Step 2: Extract RegulonsAUC for tumor cells
        regulons_auc = np.array(lf.ca['RegulonsAUC'].tolist())  # Shape: (num_cells, num_regulons)
        regulons_auc_tumor = regulons_auc[tumor_cell_indices, :]  # Filter for tumor cells

        # Step 3: Extract regulon names from the column names of RegulonsAUC
        regulon_names = lf.ca['RegulonsAUC'].dtype.names  # Extract column names (regulon names)

        # Step 4: Create a DataFrame with regulons as rows and tumor cells as columns
        self.regulons_auc_df = pd.DataFrame(regulons_auc_tumor.T, index=regulon_names, columns=tumor_cell_ids)

        # Step 5: Add a "target_genes" column
        regulons_binary_matrix = np.array(lf.ra['Regulons'].tolist())  # Shape: (num_genes, num_regulons)

        # Create a dictionary to map regulons to their target genes
        regulon_to_genes = {}
        for i, regulon in enumerate(regulon_names):
            target_gene_indices = np.where(regulons_binary_matrix[:, i] == 1)[0]
            target_genes = lf.ra['Gene'][target_gene_indices]
            regulon_to_genes[regulon] = list(target_genes)

        # Add the "target_genes" column to the DataFrame
        self.regulons_auc_df['target_genes'] = self.regulons_auc_df.index.map(regulon_to_genes)

        # Step 6: Close the .loom file
        lf.close()

        # Compute the average activity across all columns except 'target_genes'
        self.regulons_auc_df['average_activity'] = self.regulons_auc_df.iloc[:, :-1].mean(axis=1)

        # Sort by 'average_activity' in descending order
        self.sorted_df = self.regulons_auc_df.sort_values(by='average_activity', ascending=False)

        # Count the number of target genes for each regulon in the sorted DataFrame
        self.sorted_df['num_target_genes'] = self.sorted_df['target_genes'].apply(len)

    def print_top_regulons(self, top_n=10):
        print(self.sorted_df.head(top_n))
        print(self.sorted_df[['average_activity', 'num_target_genes']].head(top_n))

    def process_cnv_and_raw_counts(self):
        # Load the CNV matrix
        cnv_matrix = pd.read_csv(self.cnv_matrix_path, sep='\t', index_col=0)
        cnv_matrix = cnv_matrix * 2 - 2

        # Load the raw count matrix
        raw_count_matrix = pd.read_csv(self.raw_count_matrix_path, sep='\t', index_col=0)

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
        z_score_matrix = filtered_raw_count_matrix.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

        # Debugging: Print dimensions of the z-score matrix
        print(f"Z-score matrix dimensions: {z_score_matrix.shape}")

        # Step 2: Ensure the genes and cells match between the two matrices
        # Find common genes between CNV matrix and z-score matrix
        common_genes = cnv_matrix.index.intersection(z_score_matrix.index)
        print(f"Number of common genes: {len(common_genes)}")

        # Find common cells between CNV matrix and z-score matrix
        common_cells = cnv_matrix.columns.intersection(z_score_matrix.columns)
        print(f"Number of common cells: {len(common_cells)}")

        # Filter both matrices to include only common genes and common cells
        cnv_matrix = cnv_matrix.loc[common_genes, common_cells]
        z_score_matrix = z_score_matrix.loc[common_genes, common_cells]

        # Debugging: Print dimensions after filtering for common genes and cells
        print(f"CNV matrix dimensions after filtering: {cnv_matrix.shape}")
        print(f"Z-score matrix dimensions after filtering: {z_score_matrix.shape}")

        # Ensure the genes and cells match between the two matrices
        assert all(cnv_matrix.index == z_score_matrix.index), "Gene names do not match!"
        assert all(cnv_matrix.columns == z_score_matrix.columns), "Cell names do not match!"

        # Step 3: Map CNV to Z-scores
        results = []
        for cell in cnv_matrix.columns:
            for gene in cnv_matrix.index:
                cnv = cnv_matrix.loc[gene, cell]
                z_score = z_score_matrix.loc[gene, cell]
                results.append({'Gene': gene, 'Cell': cell, 'CNV': cnv, 'Z-score': z_score})

        self.results_df = pd.DataFrame(results)

        # Debugging: Print the first few rows of the results dataframe
        print("First few rows of the results dataframe:")
        print(self.results_df.head())

        # Step 4: Perform statistical testing (Kruskal-Wallis test)
        cnv_groups = [self.results_df[self.results_df['CNV'] == cnv]['Z-score'] for cnv in
                      self.results_df['CNV'].unique()]
        statistic, p_value = kruskal(*cnv_groups)
        print(f"Kruskal-Wallis Test Results: Statistic = {statistic}, p-value = {p_value}")

    def plot_cnv_zscore_boxplot(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            x='CNV',
            y='Z-score',
            data=self.results_df,
            palette='coolwarm',  # Use Viridis color palette
            showfliers=False  # Hide outlier dots
        )
        plt.title('Z-scores of Gene Expression per Copy Number Variation')
        plt.xlabel('Copy Number Variation (CNV)')
        plt.ylabel('Z-score of Gene Expression')

        # Save as a file (e.g., PNG, PDF)
        plt.savefig('zscore_cnv_boxplot.png', dpi=300, bbox_inches='tight')  # Save as PNG
        plt.close()  # Close the plot to free memory

    def plot_cnv_zscore_violinplot(self):
        plt.figure(figsize=(10, 6))
        ax = sns.violinplot(
            x='CNV',
            y='Z-score',
            data=self.results_df,
            palette='coolwarm',  # Use Viridis color palette
            inner='quartile',  # Show quartiles inside the violin plot
            gridsize=1000  # Increase the number of grid points for smoother violins
        )

        # Set y-axis limits
        ax.set_ylim(-2, 3)

        plt.title('Z-scores of Gene Expression per Copy Number Variation (Violin Plot)')
        plt.xlabel('Copy Number Variation (CNV)')
        plt.ylabel('Z-score of Gene Expression')

        # Save as a file (e.g., PNG, PDF)
        plt.savefig('zscore_cnv_violinplot.png', dpi=300, bbox_inches='tight')  # Save as PNG
        plt.close()  # Close the plot to free memory

    def plot_regulon_activity_vs_copy_number(self):
        # Load the CNV matrix
        cnv_matrix = pd.read_csv(self.cnv_matrix_path, sep='\t', index_col=0)

        # Step 1: Calculate the average activity of each regulon (ignore the target_genes column)
        regulons_avg_activity = self.regulons_auc_df.drop(['target_genes', 'average_activity'], axis=1).mean(axis=1)

        # Step 2: Extract the target genes for all regulons
        all_target_genes = self.regulons_auc_df['target_genes']

        # Step 3: Calculate the average copy number and standard deviation of the target genes for all regulons
        average_copy_numbers = []
        copy_number_std_devs = []  # To store standard deviations

        for target_genes in all_target_genes:
            target_genes_list = [gene.strip() for gene in target_genes if gene.strip() in cnv_matrix.index]
            if target_genes_list:
                average_copy_number = cnv_matrix.loc[target_genes_list].mean().mean()
                std_dev = cnv_matrix.loc[target_genes_list].mean().std()  # Standard deviation across genes
            else:
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
            palette='coolwarm',
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
        plt.legend([], [], frameon=False)

        # Adjust layout and display
        plt.tight_layout()
        plt.savefig('regulon_activity_cnv_scatterplot.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_all_analyses(self):
        """Run all analysis steps in sequence"""
        self.load_and_process_loom()
        self.print_top_regulons()
        self.process_cnv_and_raw_counts()
        self.plot_cnv_zscore_boxplot()
        self.plot_cnv_zscore_violinplot()
        self.plot_regulon_activity_vs_copy_number()


# Example usage:
# Example usage with automated path generation and aliquot loop
if __name__ == "__main__":
    # Base directories
    base_dir = "/work/project/ladcol_020"
    dataset_id = "ccRCC_GBM"

    # List of aliquots to process
    aliquots = [
        "C3L-00004-T1_CPT0001540013",
        "C3N-00495-T1_CPT0078510004",
        "C3L-00917-T1_CPT0023690004",
        "C3L-00088-T1_CPT0000870003",
        "C3L-00026-T1_CPT0001500003"
    ]

    # Process each aliquot
    for aliquot in aliquots:
        print(f"\n{'=' * 50}")
        print(f"Processing aliquot: {aliquot}")
        print(f"{'=' * 50}")

        # Generate paths automatically
        loom_file_path = os.path.join(base_dir, "scGRNi/RNA/SCENIC", dataset_id, aliquot, "pyscenic_output.loom")
        cnv_matrix_path = os.path.join(base_dir, "integration_visualization", dataset_id, aliquot, "cnv_matrix.tsv")
        raw_count_path = os.path.join(base_dir, "integration_visualization", dataset_id, aliquot,
                                      "raw_count_matrix.txt")

        # Verify files exist before processing
        files_to_check = [
            (loom_file_path, "Loom file"),
            (cnv_matrix_path, "CNV matrix"),
            (raw_count_path, "Raw count matrix")
        ]

        skip_aliquot = False
        for file_path, file_desc in files_to_check:
            if not os.path.exists(file_path):
                print(f"Warning: {file_desc} not found at {file_path}")
                skip_aliquot = True

        if skip_aliquot:
            print(f"Skipping analysis for {aliquot} due to missing files")
            continue

        # Create and run analysis
        try:
            analysis = RegulonAnalysis(loom_file_path, cnv_matrix_path, raw_count_path)
            analysis.run_all_analyses()
            print(f"Successfully completed analysis for {aliquot}")
        except Exception as e:
            print(f"Error processing {aliquot}: {str(e)}")