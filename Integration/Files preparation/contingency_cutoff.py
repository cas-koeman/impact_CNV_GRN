import os
import warnings
from typing import Dict, List

import loompy as lp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency


class CNVThresholdAnalyzer:
    """Analyzes CNV-regulon relationships at different thresholds for determining CNV status."""

    def __init__(self, base_dir: str = "/work/project/ladcol_020"):
        """Initialize with base directory path."""
        self.base_dir = base_dir
        self.results = {}  # Store results for all samples and thresholds

    def load_loom_data(self, dataset_id: str, aliquot: str) -> dict:
        """Load and process SCENIC loom file."""
        loom_path = os.path.join(
            self.base_dir, "scGRNi/RNA/SCENIC",
            dataset_id, aliquot, "pyscenic_output.loom"
        )

        if not os.path.exists(loom_path):
            raise FileNotFoundError(f"Loom file not found at {loom_path}")

        with lp.connect(loom_path, mode='r', validate=False) as lf:
            loom_data = {
                'regulons_auc': np.array(lf.ca['RegulonsAUC'].tolist()),
                'regulon_names': lf.ca['RegulonsAUC'].dtype.names,
                'cell_ids': lf.ca['CellID'],
                'tumor_cell_indices': np.where(lf.ca['cell_type.harmonized.cancer'] == "Tumor")[0],
                'target_genes': self._extract_target_genes(lf),
                'all_cell_types': lf.ca['cell_type.harmonized.cancer']
            }
        return loom_data

    def _extract_target_genes(self, loom_connection) -> Dict[str, list]:
        """Extract target genes mapping from loom file."""
        regulons_binary = np.array(loom_connection.ra['Regulons'].tolist())
        return {
            name: list(loom_connection.ra['Gene'][np.where(regulons_binary[:, i] == 1)[0]])
            for i, name in enumerate(loom_connection.ca['RegulonsAUC'].dtype.names)
        }

    def load_cnv_matrix(self, dataset_id: str, aliquot: str) -> pd.DataFrame:
        """Load and transform CNV matrix."""
        cnv_path = os.path.join(
            self.base_dir, "integration_GRN_CNV",
            dataset_id, aliquot, "extended_cnv_matrix.tsv"
        )

        if not os.path.exists(cnv_path):
            raise FileNotFoundError(f"CNV matrix not found at {cnv_path}")

        return pd.read_csv(cnv_path, sep='\t', index_col=0) * 2 - 2

    def _get_cnv_status(self, gene: str, cnv_matrix: pd.DataFrame, threshold: float) -> str:
        """
        Determine CNV status for a single gene across cells using a specified threshold.

        Args:
            gene: Gene name to check
            cnv_matrix: DataFrame of CNV values
            threshold: Threshold percentage for determining gain/loss

        Returns:
            'Gain', 'Loss', or 'Neutral' based on CNV status
        """
        if gene not in cnv_matrix.index:
            return "Neutral"  # Gene not in CNV matrix

        cnv_values = cnv_matrix.loc[gene]

        # Calculate percentage of cells with gains/losses
        gain_percent = (cnv_values > 0).mean()
        loss_percent = (cnv_values < 0).mean()

        if gain_percent >= threshold:
            return "Gain"
        elif loss_percent >= threshold:
            return "Loss"
        else:
            return "Neutral"

    def analyze_sample(self, dataset_id: str, aliquot: str, cell_population: str = "Tumor") -> Dict[float, Dict]:
        """
        Analyze CNV-regulon relationships for all thresholds for a single sample.

        Args:
            dataset_id: Dataset identifier
            aliquot: Sample aliquot ID
            cell_population: Cell population to analyze

        Returns:
            Dictionary with threshold values as keys and analysis results as values
        """
        print(f"\nAnalyzing sample: {aliquot}")
        print("========================================")

        # Load data
        loom_data = self.load_loom_data(dataset_id, aliquot)
        cnv_matrix = self.load_cnv_matrix(dataset_id, aliquot)

        # Check if cell population exists
        cell_types = loom_data['all_cell_types']
        pop_idx = np.where(cell_types == cell_population)[0]

        if len(pop_idx) == 0:
            print(f"Warning: Cell population '{cell_population}' not found in {aliquot}")
            return {}

        print(f"Found {len(pop_idx)} {cell_population} cells")

        # Extract AUC matrix
        auc_matrix = loom_data['regulons_auc'][pop_idx, :]

        # Create threshold results dictionary
        threshold_results = {}

        # Loop through thresholds
        for threshold in np.arange(0, 1.01, 0.1):
            # Round to handle floating point precision issues
            threshold = round(threshold, 1)
            print(f"\n  Analyzing threshold: {threshold}")

            # Analyze with current threshold
            results = self._analyze_threshold(
                auc_matrix,
                loom_data['regulon_names'],
                loom_data['target_genes'],
                cnv_matrix,
                threshold
            )

            threshold_results[threshold] = results

        return threshold_results

    def _analyze_threshold(
            self,
            auc_matrix: np.ndarray,
            regulon_names: List[str],
            target_genes: Dict[str, List[str]],
            cnv_matrix: pd.DataFrame,
            threshold: float
    ) -> Dict:
        """
        Analyze CNV-regulon relationships for a specific threshold.

        Returns:
            Dictionary with analysis results
        """
        # Initialize contingency table with zeros
        contingency_table = pd.DataFrame(
            0,
            index=['TF Loss', 'TF Neutral', 'TF Gain'],
            columns=['Target Loss', 'Target Neutral', 'Target Gain']
        )

        # Dictionary to store TF CNV status counts
        tf_cnv_counts = {'Loss': 0, 'Neutral': 0, 'Gain': 0}

        # Counters for tracking
        total_regulons = 0
        regulons_with_tf_in_cnv = 0

        for regulon in regulon_names:
            total_regulons += 1
            try:
                # Extract TF name from regulon and remove (+) suffix
                tf_name = regulon.split('_')[0].replace('(+)', '')

                # Skip this regulon if TF is not in CNV matrix
                if tf_name not in cnv_matrix.index:
                    continue

                regulons_with_tf_in_cnv += 1

                # Get CNV status for TF using current threshold
                tf_status = self._get_cnv_status(tf_name, cnv_matrix, threshold)

                # Update TF CNV counts
                tf_cnv_counts[tf_status] += 1

                # Calculate CNV status for target genes
                target_genes_list = target_genes.get(regulon, [])
                for gene in target_genes_list:
                    # Get CNV status for this target gene
                    target_status = self._get_cnv_status(gene, cnv_matrix, threshold)

                    # Update contingency table for each target gene
                    row_name = f"TF {tf_status}"
                    col_name = f"Target {target_status}"
                    contingency_table.loc[row_name, col_name] += 1

            except Exception as e:
                continue

        # Print the contingency table for this threshold
        print(f"\nContingency Table for threshold {threshold}:")
        print("--------------------------------")
        print(contingency_table)
        print("\nTF CNV Status Counts:")
        print(tf_cnv_counts)
        print("--------------------------------")

        # Add very small number to any cells that are 0 (but keep original counts)
        # Only if at least one cell has a count > 0
        if contingency_table.sum().sum() > 0:
            contingency_table = contingency_table.replace(0, 1e-10)

        # Calculate chi-square test if table has any counts
        chi2_result = None
        if contingency_table.sum().sum() > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    chi2, pval, dof, expected = chi2_contingency(
                        contingency_table,
                        lambda_="log-likelihood"
                    )
                    chi2_result = {
                        'chi2_statistic': chi2,
                        'p_value': pval,
                        'degrees_of_freedom': dof,
                        'expected_frequencies': expected
                    }
                    print(f"Chi-square p-value: {pval:.4f}")
                except Exception:
                    chi2_result = {
                        'chi2_statistic': None,
                        'p_value': None,
                        'degrees_of_freedom': None,
                        'expected_frequencies': None
                    }

        return {
            'contingency_table': contingency_table,
            'chi2_result': chi2_result,
            'tf_cnv_counts': tf_cnv_counts,
            'total_regulons': total_regulons,
            'regulons_with_tf_in_cnv': regulons_with_tf_in_cnv
        }

    def analyze_all_samples(self, dataset_id: str, sample_ids: List[str], cell_population: str = "Tumor") -> None:
        """
        Analyze all samples across all thresholds.

        Args:
            dataset_id: Dataset identifier
            sample_ids: List of sample aliquot IDs
            cell_population: Cell population to analyze
        """
        for sample in sample_ids:
            try:
                self.results[sample] = self.analyze_sample(dataset_id, sample, cell_population)
            except Exception as e:
                print(f"Failed to analyze sample {sample}: {str(e)}")
                self.results[sample] = {}

    def create_summary_table(self) -> pd.DataFrame:
        """
        Create a summary table with p-values for all samples and thresholds.

        Returns:
            DataFrame with thresholds as rows and samples as columns
        """
        # Create table with thresholds as index and samples as columns
        thresholds = np.arange(0, 1.01, 0.1)
        thresholds = [round(t, 1) for t in thresholds]  # Round to handle floating point issues

        samples = list(self.results.keys())
        summary_table = pd.DataFrame(index=thresholds, columns=samples)

        # Fill table with p-values
        for sample in samples:
            for threshold in thresholds:
                if threshold in self.results[sample]:
                    result = self.results[sample][threshold]
                    if result and result['chi2_result'] and result['chi2_result']['p_value'] is not None:
                        summary_table.loc[threshold, sample] = result['chi2_result']['p_value']
                    else:
                        summary_table.loc[threshold, sample] = None

        return summary_table

    def save_summary_table(self, file_path: str = "cnv_threshold_summary.csv") -> None:
        """Save summary table to CSV file."""
        table = self.create_summary_table()
        table.to_csv(file_path)
        print(f"Summary table saved to {file_path}")

    def plot_pvalue_boxplot(self, csv_file: str, output_file: str = None) -> None:
        """
        Create a boxplot of p-values per cutoff from a CSV file.

        Args:
            csv_file: Path to the CSV file containing p-values
            output_file: Optional path to save the plot (if None, will display)
        """
        # Read the CSV file
        df = pd.read_csv(csv_file, index_col=0)

        # Skip the first row (threshold 0.0)
        df = df.iloc[1:]

        # Convert cutoff index to percentages (0.1 -> 10%)
        df.index = [f"{float(x) * 100:.0f}%" for x in df.index]

        # Transpose the dataframe to have cutoffs as columns
        df = df.T

        # Melt the dataframe for seaborn plotting
        df_melted = df.reset_index().melt(id_vars='index', var_name='Cutoff', value_name='p-value')
        df_melted = df_melted.rename(columns={'index': 'Sample'})

        # Set style
        sns.set_style("ticks")

        # Create the boxplot
        ax = sns.violinplot(data=df_melted, x='Cutoff', y='p-value', color="white",
                            width=0.6, linewidth=1, showfliers=False)

        # Add jittered points to show individual samples with viridis colormap
        palette = sns.color_palette("icefire", n_colors=len(df_melted['Sample'].unique()))
        sns.stripplot(data=df_melted, x='Cutoff', y='p-value',
                      hue='Sample', palette=palette, alpha=0.5, jitter=0.2, size=5)

        # Customize the plot
        plt.xlabel('Threshold of cells with CNV (%)', fontsize=12)
        plt.ylabel('p-value', fontsize=12)

        # Remove top and right borders
        sns.despine(offset=10, trim=True)

        # Add significance threshold line
        plt.axhline(y=0.03, color='black', linestyle='--', linewidth=0.7, alpha=0.3)

        # Adjust layout
        plt.tight_layout()
        plt.legend().remove()

        # Save or show the plot
        if output_file:
            plt.savefig(output_file,
                        dpi=300,
                        bbox_inches='tight',
                        facecolor='white')
            print(f"Plot saved to {output_file}")
        else:
            plt.show()


def main():
    """Run CNV-regulon threshold analysis pipeline."""
    # Configuration
    dataset_id = "ccRCC_GBM"
    cell_population = "Tumor"  # The cell population to analyze

    sample_ids = [
        "C3L-00004-T1_CPT0001540013",
        "C3L-00026-T1_CPT0001500003",
        "C3L-00088-T1_CPT0000870003",
        # "C3L-00096-T1_CPT0001180011",  # Commented out as in original
        "C3L-00416-T2_CPT0010100001",
        "C3L-00448-T1_CPT0010160004",
        "C3L-00917-T1_CPT0023690004",
        "C3L-01313-T1_CPT0086820004",
        "C3N-00317-T1_CPT0012280004",
        "C3N-00495-T1_CPT0078510004"
    ]

    # Initialize analyzer
    analyzer = CNVThresholdAnalyzer()

    # Analyze all samples
    print(f"Starting CNV threshold analysis for {len(sample_ids)} samples...")
    analyzer.analyze_all_samples(dataset_id, sample_ids, cell_population)

    # Create and save summary table
    print("\nCreating summary table...")
    summary_table = analyzer.create_summary_table()
    print(summary_table)
    analyzer.save_summary_table("cnv_threshold_summary.csv")

    # Create and save boxplot
    print("\nCreating boxplot visualization...")
    analyzer.plot_pvalue_boxplot("cnv_threshold_summary.csv", "pvalue_boxplot_clean.png")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
