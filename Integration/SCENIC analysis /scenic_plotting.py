import os
from typing import Dict, Tuple, Optional

import loompy as lp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore


class SCENICDataLoader:
    """Handles all data loading and preprocessing for SCENIC analysis.

    Attributes:
        base_dir (str): Base directory path for data files
        loom_data (dict): Loaded loom file data
        expression_matrix (pd.DataFrame): Filtered expression matrix
        cnv_matrix (pd.DataFrame): Processed CNV matrix
    """

    def __init__(self, base_dir: str = "/work/project/ladcol_020"):
        """Initialize with base directory path."""
        self.base_dir = base_dir
        self.loom_data: Optional[dict] = None
        self.expression_matrix: Optional[pd.DataFrame] = None
        self.cnv_matrix: Optional[pd.DataFrame] = None

    def load_all_data(self, dataset_id: str, aliquot: str) -> None:
        """Load all required data for analysis.

        Args:
            dataset_id: Dataset identifier (e.g. "ccRCC_GBM")
            aliquot: Sample aliquot ID
        """
        self.load_loom_data(dataset_id, aliquot)
        self.load_expression_matrix(dataset_id, aliquot)
        self.load_cnv_matrix(dataset_id, aliquot)

    def load_loom_data(self, dataset_id: str, aliquot: str) -> dict:
        """Load and process SCENIC loom file.

        Returns:
            Dictionary containing:
            - regulons_auc: AUC matrix
            - regulon_names: List of regulon names
            - cell_ids: List of cell IDs
            - target_genes: Dict mapping regulons to target genes
            - tumor_cell_indices: Indices of tumor cells
        """
        loom_path = os.path.join(
            self.base_dir, "scGRNi/RNA/SCENIC",
            dataset_id, aliquot, "pyscenic_output.loom"
        )

        if not os.path.exists(loom_path):
            raise FileNotFoundError(f"Loom file not found at {loom_path}")

        with lp.connect(loom_path, mode='r', validate=False) as lf:
            self.loom_data = {
                'regulons_auc': np.array(lf.ca['RegulonsAUC'].tolist()),
                'regulon_names': lf.ca['RegulonsAUC'].dtype.names,
                'cell_ids': lf.ca['CellID'],
                'tumor_cell_indices': np.where(lf.ca['cell_type.harmonized.cancer'] == "Tumor")[0],
                'target_genes': self._extract_target_genes(lf),
                'all_cell_types': lf.ca['cell_type.harmonized.cancer']
            }
        return self.loom_data

    def _extract_target_genes(self, loom_connection) -> Dict[str, list]:
        """Extract target genes mapping from loom file."""
        regulons_binary = np.array(loom_connection.ra['Regulons'].tolist())
        return {
            name: list(loom_connection.ra['Gene'][np.where(regulons_binary[:, i] == 1)[0]])
            for i, name in enumerate(loom_connection.ca['RegulonsAUC'].dtype.names)
        }

    def load_expression_matrix(self, dataset_id: str, aliquot: str) -> pd.DataFrame:
        """Load and filter expression matrix."""
        expr_path = os.path.join(
            self.base_dir, "integration_GRN_CNV",
            dataset_id, aliquot, "raw_count_matrix.txt"
        )

        if not os.path.exists(expr_path):
            raise FileNotFoundError(f"Expression matrix not found at {expr_path}")

        df = pd.read_csv(expr_path, sep='\t', index_col=0)

        # Apply quality filters
        gene_filter = (df > 0).sum(axis=1) >= 200
        cell_filter = (df.loc[gene_filter] > 0).sum(axis=0) >= 3

        self.expression_matrix = df.loc[gene_filter, cell_filter]
        return self.expression_matrix

    def load_cnv_matrix(self, dataset_id: str, aliquot: str) -> pd.DataFrame:
        """Load and transform CNV matrix."""
        cnv_path = os.path.join(
            self.base_dir, "integration_GRN_CNV",
            dataset_id, aliquot, "extended_cnv_matrix.tsv"
        )

        if not os.path.exists(cnv_path):
            raise FileNotFoundError(f"CNV matrix not found at {cnv_path}")

        self.cnv_matrix = pd.read_csv(cnv_path, sep='\t', index_col=0) * 2 - 2
        return self.cnv_matrix

    def get_common_features(self) -> Tuple[pd.Index, pd.Index]:
        """Get intersection of genes and cells across loaded data."""
        if self.expression_matrix is None or self.cnv_matrix is None:
            raise ValueError("Must load both expression and CNV data first")

        common_genes = self.expression_matrix.index.intersection(
            self.cnv_matrix.index
        )
        common_cells = self.expression_matrix.columns.intersection(
            self.cnv_matrix.columns
        )

        return common_genes, common_cells


class SCENICPlotter:
    """Handles all visualization tasks for SCENIC data."""

    def __init__(self, data_loader: SCENICDataLoader):
        """Initialize with data loader and output directory."""
        self.loader = data_loader

        # Processed data containers
        self.regulons_auc_df: Optional[pd.DataFrame] = None
        self.sorted_regulons: Optional[pd.DataFrame] = None

    def process_regulon_data(self) -> None:
        """Process loom data into analysis-ready DataFrames."""
        if self.loader.loom_data is None:
            raise ValueError("No loom data loaded")

        loom_data = self.loader.loom_data
        tumor_ids = loom_data['cell_ids'][loom_data['tumor_cell_indices']]
        auc_matrix = loom_data['regulons_auc'][loom_data['tumor_cell_indices'], :]

        self.regulons_auc_df = pd.DataFrame(
            auc_matrix.T,
            index=loom_data['regulon_names'],
            columns=tumor_ids
        )
        self.regulons_auc_df['target_genes'] = self.regulons_auc_df.index.map(
            loom_data['target_genes']
        )

        # Calculate average activity
        self.regulons_auc_df['average_activity'] = \
            self.regulons_auc_df.iloc[:, :-1].mean(axis=1)

        # Sort regulons by activity
        self.sorted_regulons = self.regulons_auc_df.sort_values(
            'average_activity', ascending=False
        )
        self.sorted_regulons['num_target_genes'] = \
            self.sorted_regulons['target_genes'].apply(len)

    def plot_regulon_activity_vs_cnv(self) -> None:
        """Plot regulon activity vs average CNV of target genes."""
        if self.regulons_auc_df is None:
            self.process_regulon_data()

        if self.loader.cnv_matrix is None:
            raise ValueError("CNV matrix not loaded")

        cnv_matrix = self.loader.cnv_matrix
        regulons_activity = self.regulons_auc_df.drop(
            ['target_genes', 'average_activity'], axis=1
        ).mean(axis=1)

        # Calculate average CNV per regulon
        avg_cnvs, std_cnvs = [], []
        for genes in self.regulons_auc_df['target_genes']:
            genes = [g.strip() for g in genes if g.strip() in cnv_matrix.index]
            if genes:
                avg_cnvs.append(cnv_matrix.loc[genes].mean().mean())
                std_cnvs.append(cnv_matrix.loc[genes].mean().std())
            else:
                avg_cnvs.append(np.nan)
                std_cnvs.append(np.nan)

        plot_data = pd.DataFrame({
            'Average_CNV': avg_cnvs,
            'CNV_StdDev': std_cnvs,
            'Regulon_Activity': regulons_activity
        }).dropna()

        # Create plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x='Average_CNV',
            y='Regulon_Activity',
            hue='Regulon_Activity',
            data=plot_data,
            palette='coolwarm',
            s=40,
            alpha=1,
            edgecolor='none'
        )

        plt.errorbar(
            x=plot_data['Average_CNV'],
            y=plot_data['Regulon_Activity'],
            xerr=plot_data['CNV_StdDev'],
            fmt='none',
            ecolor='gray',
            alpha=0.4
        )

        # Label top regulons
        top_regulons = plot_data.nlargest(5, 'Regulon_Activity')
        for regulon, row in top_regulons.iterrows():
            plt.text(
                row['Average_CNV'],
                row['Regulon_Activity'] + 0.001,
                regulon,
                fontsize=10,
                ha='center'
            )

        plt.xlabel('Average CNV of Target Genes')
        plt.ylabel('Average Regulon Activity')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(
            os.path.join('regulon_activity_vs_cnv.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()

    def plot_cnv_zscore_distribution(self) -> None:
        """Plot distribution of z-scores by CNV status."""
        if self.loader.expression_matrix is None or self.loader.cnv_matrix is None:
            raise ValueError("Expression and CNV data must be loaded first")

        common_genes, common_cells = self.loader.get_common_features()
        z_scores = self.loader.expression_matrix.loc[common_genes, common_cells].apply(
            zscore, axis=1
        )
        cnv_data = self.loader.cnv_matrix.loc[common_genes, common_cells]

        # Combine into long format
        results = []
        for cell in common_cells:
            for gene in common_genes:
                results.append({
                    'Gene': gene,
                    'Cell': cell,
                    'CNV': cnv_data.loc[gene, cell],
                    'Z-score': z_scores.loc[gene, cell]
                })

        results_df = pd.DataFrame(results)
        results_df['CNV_Status'] = results_df['CNV'].apply(
            lambda x: 'Loss' if x < 0 else ('Gain' if x > 0 else 'Neutral')
        )

        # Create plots
        plt.figure(figsize=(12, 5))

        plt.subplot(121)
        sns.boxplot(
            x='CNV_Status',
            y='Z-score',
            data=results_df,
            order=['Loss', 'Neutral', 'Gain']
        )
        plt.title('Z-score Distribution by CNV Status')

        plt.subplot(122)
        sns.violinplot(
            x='CNV_Status',
            y='Z-score',
            data=results_df,
            order=['Loss', 'Neutral', 'Gain']
        )
        plt.title('Z-score Density by CNV Status')

        plt.tight_layout()
        plt.savefig(
            os.path.join('cnv_zscore_distributions.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()


def main():
    """Run code for plotting the SCENIC regulon results"""
    # Configuration
    dataset_id = "ccRCC_GBM"

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

    for sample in sample_ids:
        try:
            # Initialize data loader
            print(f"\nLoading data for sample: {sample}")
            loader = SCENICDataLoader()
            loader.load_all_data(dataset_id, sample)

            # Plot the results
            plotter = SCENICPlotter()
            plotter.plot_regulon_activity_vs_cnv()
            plotter.plot_cnv_zscore_distribution()

        except Exception as e:
            print(f"Analysis failed for sample {sample}: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
