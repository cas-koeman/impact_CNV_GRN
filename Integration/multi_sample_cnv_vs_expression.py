import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


class MultiSampleCNVExpressionAnalyzer:
    def __init__(self, datasets, cnv_dir, raw_count_dir):
        """
        Initialize the class with dataset names and directory paths.

        Args:
            datasets (list): List of dataset names.
            cnv_dir (str): Directory for CNV matrix files.
            raw_count_dir (str): Directory for raw count matrix files.
        """
        self.datasets = datasets
        self.cnv_dir = cnv_dir
        self.raw_count_dir = raw_count_dir
        self.aggregated_results = []

    def _process_single_dataset(self, dataset):
        """
        Process a single dataset: filter raw counts, normalize, match genes and cells, and calculate mean z-scores per CNV state.

        Args:
            dataset (str): Name of the dataset.

        Returns:
            pd.DataFrame: Aggregated results DataFrame for the dataset.
        """
        print(f"\nProcessing dataset: {dataset}")

        # Define file paths
        cnv_matrix_path = os.path.join(self.cnv_dir, dataset, "cnv_matrix.tsv")
        raw_count_path = os.path.join(self.raw_count_dir, dataset, "raw_count_matrix.txt")

        print(f"Loading CNV matrix from: {cnv_matrix_path}")
        cnv_matrix = pd.read_csv(cnv_matrix_path, sep='\t', index_col=0)
        cnv_matrix = cnv_matrix * 2 - 2  # Adjust CNV values
        print(f"CNV matrix shape: {cnv_matrix.shape}")

        print(f"Loading raw count matrix from: {raw_count_path}")
        raw_count_matrix = pd.read_csv(raw_count_path, sep='\t', index_col=0)
        print(f"Raw count matrix shape: {raw_count_matrix.shape}")

        # Step 1: Filter the raw count matrix
        print("Filtering raw count matrix...")
        gene_filter = (raw_count_matrix > 0).sum(axis=1) >= 200  # Genes expressed in >= 200 cells
        filtered_raw_count_matrix = raw_count_matrix[gene_filter]

        cell_filter = (filtered_raw_count_matrix > 0).sum(axis=0) >= 3  # Cells expressing >= 3 genes
        filtered_raw_count_matrix = filtered_raw_count_matrix.loc[:, cell_filter]
        print(f"Filtered raw count matrix shape: {filtered_raw_count_matrix.shape}")

        # Step 2: Z-normalize the filtered raw count matrix
        print("Z-normalizing the filtered raw count matrix...")
        z_score_matrix = filtered_raw_count_matrix.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        print(f"Z-score matrix shape: {z_score_matrix.shape}")

        # Step 3: Match genes and cells between CNV and z-score matrices
        print("Matching genes and cells between CNV and Z-score matrices...")
        common_genes = cnv_matrix.index.intersection(z_score_matrix.index)
        common_cells = cnv_matrix.columns.intersection(z_score_matrix.columns)

        # Filter both matrices to include only common genes and common cells
        cnv_matrix = cnv_matrix.loc[common_genes, common_cells]
        z_score_matrix = z_score_matrix.loc[common_genes, common_cells]
        print(f"Common genes: {len(common_genes)}, Common cells: {len(common_cells)}")

        # Step 4: Calculate mean z-scores per CNV state
        print("Calculating mean z-scores per CNV state...")
        results = []
        # Use np.unique() instead of .unique() for NumPy arrays
        for cnv_state in np.unique(cnv_matrix.values.flatten().astype(float).round(4)):
            # Get mask of where this CNV state occurs
            mask = cnv_matrix == cnv_state

            # Get z-scores for this CNV state
            z_scores = z_score_matrix.values[mask.values]

            # Calculate mean z-score if there are values
            if len(z_scores) > 0:
                mean_z_score = np.mean(z_scores)
                results.append({'Dataset': dataset, 'CNV': cnv_state, 'Mean Z-score': mean_z_score})
                print(f"CNV state {cnv_state}: Mean Z-score = {mean_z_score}")

        return pd.DataFrame(results)

    def process_all_datasets(self):
        """
        Process all datasets and aggregate results into a single DataFrame.
        """
        print("\nStarting to process all datasets...")
        for dataset in self.datasets:
            print(f"\nProcessing dataset: {dataset}")
            results_df = self._process_single_dataset(dataset)
            self.aggregated_results.append(results_df)
            print(f"Finished processing dataset: {dataset}")

        # Combine results from all datasets into a single DataFrame
        print("\nCombining results from all datasets...")
        self.aggregated_results_df = pd.concat(self.aggregated_results, ignore_index=True)
        print("All datasets processed and results aggregated.")

    def plot_aggregated_scatter(self):
        """
        Create a scatterplot of mean Z-scores vs CNV for all datasets, with jitter and a linear trendline.
        """
        print("\nCreating scatterplot with trendline...")
        plt.figure(figsize=(12, 8))

        # Filter data to include only CNV values between -2 and 2
        filtered_df = self.aggregated_results_df[
            (self.aggregated_results_df['CNV'] >= -2) & (self.aggregated_results_df['CNV'] <= 2)
            ]

        # Define a jitter function to add random noise
        def jitter(values, j):
            return values + np.random.normal(j, 0.1, values.shape)

        # Add jitter to the x-axis
        filtered_df['CNV_jittered'] = jitter(filtered_df['CNV'].astype(float), 0.1)

        # Create the scatterplot
        ax = sns.scatterplot(
            x='CNV_jittered',
            y='Mean Z-score',
            hue='Dataset',  # Color by dataset
            data=filtered_df,
            palette='viridis',  # Use a color palette
            s=100,  # Adjust the size of the dots
            alpha=0.6  # Add transparency
        )

        # Add linear regression line
        x = filtered_df['CNV'].astype(float)
        y = filtered_df['Mean Z-score']

        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2

        # Get x values for the line
        x_line = np.linspace(-2, 2, 100)
        y_line = slope * x_line + intercept

        # Plot the regression line (without adding to legend)
        plt.plot(x_line, y_line, color='black', linewidth=0.5, linestyle="--", label='_nolegend_')

        # Add R² value and regression equation
        plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes,
                 fontsize=18, verticalalignment='top')

        equation = f'y = {slope:.4f}x + {intercept:.4f}'
        plt.text(0.05, 0.90, equation, transform=plt.gca().transAxes,
                 fontsize=18, verticalalignment='top')

        # Set x-axis limits and ticks
        plt.xlim(-2.5, 2.5)
        plt.xticks([-2, -1, 0, 1, 2])  # Explicitly set the x-axis ticks

        # Remove top and right spines
        sns.despine()

        plt.title('Mean Z-scores of Gene Expression per Copy Number Variation (All Datasets)')
        plt.xlabel('Copy Number Variation (CNV)')
        plt.ylabel('Mean Z-score of Gene Expression')

        # Move legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig('zscore_cnv_scatter_all_datasets.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Scatterplot with trendline saved as 'zscore_cnv_scatter_all_datasets.png'.")


# Example usage:
sample_ids = [
    "C3L-00004-T1_CPT0001540013",
    "C3L-00026-T1_CPT0001500003",
    "C3L-00088-T1_CPT0000870003",
    # "C3L-00096-T1_CPT0001180011",
    "C3L-00416-T2_CPT0010100001",
    "C3L-00448-T1_CPT0010160004",
    "C3L-00917-T1_CPT0023690004",
    "C3L-01313-T1_CPT0086820004",
    "C3N-00317-T1_CPT0012280004",
    "C3N-00495-T1_CPT0078510004"
]

cnv_dir = "/work/project/ladcol_020/integration_GRN_CNV/ccRCC_GBM/"
raw_count_dir = "/work/project/ladcol_020/integration_GRN_CNV/ccRCC_GBM/"

# Initialize and run the multi-dataset analysis
print("Initializing MultiDatasetCNVAnalysis...")
multi_analysis = MultiSampleCNVExpressionAnalyzer(datasets, cnv_dir, raw_count_dir)

print("\nProcessing all datasets...")
multi_analysis.process_all_datasets()

print("\nPlotting aggregated scatter plot...")
multi_analysis.plot_aggregated_scatter()
print("Analysis complete!")
