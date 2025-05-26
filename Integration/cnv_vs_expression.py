import os
from typing import List, Optional

import loompy as lp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal


class CNVExpressionAnalyzer:
    """
    Analyzes the relationship between copy number variations (CNVs) and gene expression.

    This class processes single-cell RNA-seq data along with CNV information to explore
    how gene expression levels correlate with copy number variations. It provides
    visualization and statistical analysis tools for these relationships.

    Attributes:
        loom_file_path (str): Path to the SCENIC output loom file
        cnv_matrix_path (str): Path to the CNV matrix file (tab-separated)
        raw_count_matrix_path (str): Path to the raw count matrix file (tab-separated)
        aliquot_name (str): Name of the current sample aliquot being analyzed
        results_df (Optional[pd.DataFrame]): DataFrame with combined CNV and Z-score data
        output_dir (str): Directory to save output files
    """

    def __init__(self,
                 loom_file_path: str,
                 cnv_matrix_path: str,
                 raw_count_matrix_path: str,
                 aliquot_name: str,
                 output_dir: Optional[str] = None):
        """
        Initialize the CNV Expression Analyzer with file paths.

        Args:
            loom_file_path: Path to the SCENIC output loom file
            cnv_matrix_path: Path to the CNV matrix file (tab-separated)
            raw_count_matrix_path: Path to the raw count matrix file (tab-separated)
            aliquot_name: Name of the current sample aliquot being analyzed
            output_dir: Directory to save output files (defaults to a standard path if None)
        """
        self.loom_file_path = loom_file_path
        self.cnv_matrix_path = cnv_matrix_path
        self.raw_count_matrix_path = raw_count_matrix_path
        self.aliquot_name = aliquot_name
        self.results_df = None

        # Set default output directory if not provided
        if output_dir is None:
            self.output_dir = os.path.join("/work/project/ladcol_020/integration_GRN_CNV/ccRCC_GBM",
                                           self.aliquot_name)
        else:
            self.output_dir = output_dir

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_expression_vs_cnv(self) -> None:
        """
        Analyze the relationship between copy number variations and gene expression.

        This function loads CNV and raw count matrices, filters them, normalizes expression,
        and combines the data for statistical analysis. It performs a Kruskal-Wallis test
        to evaluate if gene expression significantly differs across CNV levels.
        """
        # Load data matrices
        cnv_matrix = pd.read_csv(self.cnv_matrix_path, sep='\t', index_col=0)
        cnv_matrix = cnv_matrix # * 2 - 2  # Convert to copy number change relative to diploid
        raw_count_matrix = pd.read_csv(self.raw_count_matrix_path, sep='\t', index_col=0)

        print(f"CNV matrix dimensions: {cnv_matrix.shape}")
        print(f"Raw count matrix dimensions (before filtering): {raw_count_matrix.shape}")

        # Filter genes and cells
        gene_filter = (raw_count_matrix > 0).sum(axis=1) >= 200
        filtered_matrix = raw_count_matrix[gene_filter]
        cell_filter = (filtered_matrix > 0).sum(axis=0) >= 3
        filtered_matrix = filtered_matrix.loc[:, cell_filter]

        print(f"Filtered raw count matrix dimensions: {filtered_matrix.shape}")

        # Z-normalize expression data
        z_score_matrix = filtered_matrix.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        print(f"Z-score matrix dimensions: {z_score_matrix.shape}")

        # Find common genes and cells
        common_genes = cnv_matrix.index.intersection(z_score_matrix.index)
        common_cells = cnv_matrix.columns.intersection(z_score_matrix.columns)
        print(f"Number of common genes: {len(common_genes)}")
        print(f"Number of common cells: {len(common_cells)}")

        # Filter matrices to common elements
        cnv_matrix = cnv_matrix.loc[common_genes, common_cells]
        z_score_matrix = z_score_matrix.loc[common_genes, common_cells]

        print(f"CNV matrix dimensions after filtering: {cnv_matrix.shape}")
        print(f"Z-score matrix dimensions after filtering: {z_score_matrix.shape}")

        # Verify alignment
        assert all(cnv_matrix.index == z_score_matrix.index), "Gene names do not match!"
        assert all(cnv_matrix.columns == z_score_matrix.columns), "Cell names do not match!"

        # Combine CNV and Z-score data
        results = []
        for cell in cnv_matrix.columns:
            for gene in cnv_matrix.index:
                cnv = cnv_matrix.loc[gene, cell]
                z_score = z_score_matrix.loc[gene, cell]
                results.append({'Gene': gene, 'Cell': cell, 'CNV': cnv, 'Z-score': z_score})

        self.results_df = pd.DataFrame(results)
        print("First few rows of the results dataframe:")
        print(self.results_df.head())

        # Statistical testing
        cnv_groups = [self.results_df[self.results_df['CNV'] == cnv]['Z-score']
                      for cnv in self.results_df['CNV'].unique()]
        statistic, p_value = kruskal(*cnv_groups)
        print(f"Kruskal-Wallis Test Results: Statistic = {statistic}, p-value = {p_value}")

    def create_boxplot(self) -> None:
        """
        Create a boxplot showing the relationship between CNV and expression Z-scores.

        The plot displays Z-score distributions for each CNV level, allowing visual
        comparison of how gene expression varies across different copy number states.
        """
        if self.results_df is None:
            raise ValueError("Results data not available. Run analyze_expression_vs_cnv first.")

        plt.figure(figsize=(16, 12))
        sns.boxplot(
            x='CNV',
            y='Z-score',
            data=self.results_df,
            palette='coolwarm',
            showfliers=False
        )
        plt.xlabel('Copy Number Variation (CNV)', fontsize=28)
        plt.ylabel('Z-score of Gene Expression', fontsize=28)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        output_file = os.path.join(self.output_dir, 'zscore_cnv_boxplot.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Boxplot saved to: {output_file}")

    def create_violinplot(self) -> None:
        """
        Create a violin plot showing the relationship between CNV and expression Z-scores.

        The plot displays Z-score distributions for each CNV level with more detail on
        the density of data points, providing insight into the expression distribution shape.
        """
        if self.results_df is None:
            raise ValueError("Results data not available. Run analyze_expression_vs_cnv first.")

        plt.figure(figsize=(16, 12))
        ax = sns.violinplot(
            x='CNV',
            y='Z-score',
            data=self.results_df,
            palette='coolwarm',
            inner='quartile',
            gridsize=1000
        )
        ax.set_ylim(-2, 3)
        plt.xlabel('Copy Number Variation (CNV)', fontsize=28)
        plt.ylabel('Z-score of Gene Expression', fontsize=28)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        output_file = os.path.join(self.output_dir, 'zscore_cnv_violinplot.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Violin plot saved to: {output_file}")

    def run_analysis_pipeline(self) -> None:
        """
        Run the complete analysis pipeline in sequence.

        This function executes all analysis steps in the proper order:
        1. Analyzing expression vs CNV relationships
        2. Creating visualizations
        """
        self.analyze_expression_vs_cnv()
        self.create_boxplot()
        self.create_violinplot()
        print(f"Analysis completed for sample: {self.aliquot_name}")


def process_dataset(base_dir: str, dataset_id: str, aliquots: List[str]) -> None:
    """
    Process a set of aliquots from a dataset.

    Args:
        base_dir: Base directory containing all data
        dataset_id: ID of the dataset being analyzed
        aliquots: List of aliquot IDs to process
    """
    for aliquot in aliquots:
        print(f"\nProcessing aliquot: {aliquot}")

        # Generate file paths
        loom_file_path = os.path.join(base_dir, "scGRNi/RNA/SCENIC", dataset_id, aliquot, "pyscenic_output.loom")
        cnv_matrix_path = os.path.join(base_dir, "integration_GRN_CNV", dataset_id, aliquot, "extended_cnv_matrix.tsv")
        raw_count_path = os.path.join(base_dir, "integration_GRN_CNV", dataset_id, aliquot,
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

        # Run analysis
        try:
            analyzer = CNVExpressionAnalyzer(loom_file_path, cnv_matrix_path, raw_count_path, aliquot)
            analyzer.run_analysis_pipeline()
            print(f"Successfully completed analysis for {aliquot}")
        except Exception as e:
            print(f"Error processing {aliquot}: {str(e)}")


if __name__ == "__main__":
    # Configuration
    BASE_DIR = "/work/project/ladcol_020"
    DATASET_ID = "ccRCC_GBM"

    # List of aliquots to process
    ALIQUOTS = [
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

    # Run processing pipeline
    process_dataset(BASE_DIR, DATASET_ID, ALIQUOTS)
