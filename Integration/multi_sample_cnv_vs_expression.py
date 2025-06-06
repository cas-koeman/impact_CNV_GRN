import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

class MultiSampleCNVExpressionAnalyzer:
    def __init__(self, samples, cnv_dir, raw_count_dir):
        """Initialize the class with dataset names and directory paths.

        Args:
            samples (list): List of dataset names.
            cnv_dir (str): Directory for CNV matrix files.
            raw_count_dir (str): Directory for raw count matrix files.
        """
        self.samples = samples
        self.cnv_dir = cnv_dir
        self.raw_count_dir = raw_count_dir
        self.aggregated_results = []

    def _process_single_dataset(self, dataset):
        """Process a single dataset: filter raw counts, normalize, match genes and cells, and calculate mean z-scores per CNV state.

        Args:
            dataset (str): Name of the dataset.

        Returns:
            pd.DataFrame: Aggregated results DataFrame for the dataset.
        """
        print(f"\nProcessing dataset: {dataset}")

        # Define file paths
        cnv_matrix_path = os.path.join(self.cnv_dir, dataset, "extended_cnv_matrix.tsv")
        count_matrix_path = os.path.join(self.raw_count_dir, dataset, "residual_matrix.txt")
        # count_matrix_path = os.path.join(self.raw_count_dir, dataset, "raw_count_matrix.txt")


        print(f"Loading CNV matrix from: {cnv_matrix_path}")
        cnv_matrix = pd.read_csv(cnv_matrix_path, sep='\t', index_col=0)
        cnv_matrix = cnv_matrix 
        print(f"CNV matrix shape: {cnv_matrix.shape}")
        print(f"Unique CNV values in {dataset}: {np.unique(cnv_matrix.values)}")

        print(f"Loading raw count matrix from: {count_matrix_path}")
        raw_count_matrix = pd.read_csv(count_matrix_path, sep='\t', index_col=0)
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
                gene_count = mask.sum().sum()  # Total number of gene-cell observations for this CNV state
                results.append({'Dataset': dataset, 'CNV': cnv_state, 'Mean Z-score': mean_z_score, 'Gene Count': gene_count})
                print(f"CNV state {cnv_state}: Mean Z-score = {mean_z_score} (from {gene_count} gene-cell observations)")

        return pd.DataFrame(results)

    def process_all_samples(self):
        """Process all samples and aggregate results into a single DataFrame."""
        print("\nStarting to process all samples...")
        for dataset in self.samples:
            print(f"\nProcessing dataset: {dataset}")
            results_df = self._process_single_dataset(dataset)
            self.aggregated_results.append(results_df)
            print(f"Finished processing dataset: {dataset}")

        # Combine results from all samples into a single DataFrame
        print("\nCombining results from all samples...")
        self.aggregated_results_df = pd.concat(self.aggregated_results, ignore_index=True)
        print("All samples processed and results aggregated.")

    def plot_aggregated_scatter(self):
        """Create a scatterplot of mean Z-scores vs CNV for all samples, with jitter and a linear trendline."""
        print("\nCreating scatterplot with trendline...")
        
        # Set consistent styling
        sns.set_style("ticks")
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 24,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 20
        })

        # Create figure
        plt.figure(figsize=(16, 9))

        # Clean up sample names by removing everything after underscore
        self.aggregated_results_df['Dataset_clean'] = self.aggregated_results_df['Dataset'].str.split('_').str[0]

        # Filter data to only include CNV values between 0.5 and 2
        filtered_df = self.aggregated_results_df[
            (self.aggregated_results_df['CNV'] >= 0.5) & 
            (self.aggregated_results_df['CNV'] <= 2)
        ]

        # Define a jitter function to add random noise
        def jitter(values, j):
            return values + np.random.normal(j, 0.05, values.shape)

        # Add jitter to the x-axis
        filtered_df['CNV_jittered'] = jitter(filtered_df['CNV'].astype(float), 0.05)

        # Define colors using coolwarm colormap
        unique_samples = filtered_df['Dataset_clean'].unique()
        sample_colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_samples)))
        color_dict = dict(zip(unique_samples, sample_colors))

        # Create the base scatterplot using cleaned names
        for sample, group in filtered_df.groupby('Dataset_clean'):
            plt.scatter(
                x=group['CNV_jittered'],
                y=group['Mean Z-score'],
                s=100,
                alpha=0.8,
                color=color_dict[sample],
                edgecolor='white',
                label=sample
            )

        # Add regression line using sns.regplot
        sns.regplot(
            x=filtered_df['CNV'].astype(float),
            y=filtered_df['Mean Z-score'],
            scatter=False,  # We already have our custom scatter points
            ci=95,
            line_kws={'color': '#cfcfcf', 'linewidth': 2.5},
            scatter_kws={'alpha': 0},  # Make scatter points invisible
            truncate=False
        )

        # Calculate and display R² value
        x = filtered_df['CNV'].astype(float)
        y = filtered_df['Mean Z-score']
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2
        print(f"R² value (0.5-2 CNV range): {r_squared:.3f}")
        print(f"Slope: {slope:.3f}, Intercept: {intercept:.3f}")

        # Add R² value to plot
        plt.text(0.8, 0.1, f'R² = {r_squared:.3f}', transform=plt.gca().transAxes,
                fontsize=18, verticalalignment='top')

        # Set x and y limits and ticks
        plt.xlim(0.4, 2.1)
        plt.ylim(-0.2, 0.5)
        plt.xticks([0.5, 1.0, 1.5, 2.0], fontsize=18)
        plt.yticks(fontsize=18)

        # Remove top and right spines
        sns.despine()

        # Axis labels with correct font size
        plt.xlabel('Copy Number Variation (CNV)', labelpad=10)
        plt.ylabel('Mean Z-score of Gene Expression', labelpad=10)

        # Move legend outside the plot and use cleaned names
        plt.legend(loc='upper left', fontsize=14, frameon=False, title='Sample', title_fontsize=18)

        plt.tight_layout(pad=3.0)
        plt.savefig('zscore_cnv_scatter_all_samples.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Scatterplot with trendline saved as 'zscore_cnv_scatter_all_samples.png'.")


# Example usage:
sample_ids = [
        "C3L-00004-T1_CPT0001540013",
        "C3L-00026-T1_CPT0001500003",
        "C3L-00088-T1_CPT0000870003",
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
multi_analysis = MultiSampleCNVExpressionAnalyzer(sample_ids, cnv_dir, raw_count_dir)

print("\nProcessing all samples...")
multi_analysis.process_all_samples()

print("\nPlotting aggregated scatter plot...")
multi_analysis.plot_aggregated_scatter()
print("Analysis complete!")