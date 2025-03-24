import loompy as lp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import kruskal, zscore, chi2_contingency
from matplotlib_venn import venn2, venn3
from statsmodels.graphics.mosaicplot import mosaic
from typing import Dict, Tuple, Optional


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
            self.base_dir, "integration_CNV_GRN",
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
            dataset_id, aliquot, "cnv_matrix.tsv"
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

    def __init__(self, data_loader: SCENICDataLoader, output_dir: str = "plots"):
        """Initialize with data loader and output directory."""
        self.loader = data_loader
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

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
            os.path.join(self.output_dir, 'regulon_activity_vs_cnv.png'),
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
            os.path.join(self.output_dir, 'cnv_zscore_distributions.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()


class SCENICComparer:
    """Handles comparison of regulons between cell types/conditions."""

    def __init__(self, data_loader: SCENICDataLoader, output_dir: str = "comparisons"):
        """Initialize with data loader and output directory."""
        self.loader = data_loader
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Analysis results
        self.contingency_tables: Dict[str, pd.DataFrame] = {}
        self.chi2_results: Dict[str, dict] = {}

    def compare_conditions(self, condition1: str, condition2: str) -> None:
        """Compare regulons between two conditions.

        Args:
            condition1: First condition name (e.g. "Tumor")
            condition2: Second condition name (e.g. "Normal")
        """
        if self.loader.loom_data is None:
            raise ValueError("No loom data loaded")

        loom_data = self.loader.loom_data
        cell_types = loom_data['all_cell_types']

        # Get indices for each condition
        cond1_idx = np.where(cell_types == condition1)[0]
        cond2_idx = np.where(cell_types == condition2)[0]

        if len(cond1_idx) == 0 or len(cond2_idx) == 0:
            raise ValueError(f"One or both conditions not found in data")

        # Extract AUC matrices for each condition
        auc_matrix = loom_data['regulons_auc']
        cond1_auc = auc_matrix[cond1_idx, :]
        cond2_auc = auc_matrix[cond2_idx, :]

        # Create DataFrames for each condition
        regulon_names = loom_data['regulon_names']
        cond1_df = pd.DataFrame(
            cond1_auc.T,
            index=regulon_names,
            columns=loom_data['cell_ids'][cond1_idx]
        )
        cond2_df = pd.DataFrame(
            cond2_auc.T,
            index=regulon_names,
            columns=loom_data['cell_ids'][cond2_idx]
        )

        # Calculate average activities
        cond1_df['avg_activity'] = cond1_df.mean(axis=1)
        cond2_df['avg_activity'] = cond2_df.mean(axis=1)

        # Plot Venn diagram of active regulons
        self._plot_venn_diagram(
            cond1_df, cond2_df,
            f"{condition1} vs {condition2} Regulons"
        )

        # Analyze CNV relationships
        self._analyze_cnv_relationships(
            cond1_df, cond2_df,
            condition1, condition2
        )

    def _plot_venn_diagram(self, df1: pd.DataFrame, df2: pd.DataFrame,
                           title: str) -> None:
        """Plot Venn diagram of significant regulons."""
        active_threshold = 0.1  # Example threshold

        set1 = set(df1[df1['avg_activity'] > active_threshold].index)
        set2 = set(df2[df2['avg_activity'] > active_threshold].index)

        plt.figure(figsize=(8, 6))
        venn2([set1, set2], set_labels=[df1.name, df2.name])
        plt.title(title)
        plt.savefig(
            os.path.join(self.output_dir, f"venn_{title.replace(' ', '_')}.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close()

    def _analyze_cnv_relationships(self, df1: pd.DataFrame, df2: pd.DataFrame,
                                   cond1_name: str, cond2_name: str) -> None:
        """Analyze CNV relationships between conditions."""
        if self.loader.cnv_matrix is None:
            raise ValueError("CNV matrix not loaded")

        target_genes = self.loader.loom_data['target_genes']
        cnv_matrix = self.loader.cnv_matrix

        # Create contingency tables
        for cond, df in [(cond1_name, df1), (cond2_name, df2)]:
            table = pd.DataFrame(
                0,
                index=['Loss', 'Neutral', 'Gain'],
                columns=['TG Loss', 'TG Neutral', 'TG Gain']
            )

            for tf in df.index:
                tf_status = self._get_cnv_status([tf], cnv_matrix)
                tg_status = self._get_cnv_status(target_genes[tf], cnv_matrix)

                for status in tg_status:
                    table.loc[tf_status, f'TG {status}'] += 1

            self.contingency_tables[cond] = table
            chi2, p, dof, expected = chi2_contingency(table)
            self.chi2_results[cond] = {
                'chi2': chi2,
                'p_value': p,
                'dof': dof,
                'expected': expected
            }

        # Plot mosaic plots
        self._plot_mosaic_comparison(cond1_name, cond2_name)

    @staticmethod
    def _get_cnv_status(genes: list, cnv_matrix: pd.DataFrame) -> str:
        """Classify CNV status for a list of genes."""
        if not genes:
            return 'Neutral'

        present_genes = [g for g in genes if g in cnv_matrix.index]
        if not present_genes:
            return 'Neutral'

        avg_cnv = cnv_matrix.loc[present_genes].mean().mean()
        return 'Loss' if avg_cnv < 0 else ('Gain' if avg_cnv > 0 else 'Neutral')

    def _plot_mosaic_comparison(self, cond1: str, cond2: str) -> None:
        """Plot mosaic plots comparing CNV relationships."""
        cmap = plt.cm.RdBu_r

        for cond in [cond1, cond2]:
            table = self.contingency_tables[cond]
            res = self.chi2_results[cond]
            residuals = (table.values - res['expected']) / np.sqrt(res['expected'])
            abs_max = np.max(np.abs(residuals))
            norm = plt.Normalize(-abs_max, abs_max)

            # Prepare data for mosaic plot
            props = table.div(table.sum(axis=1), axis=0)
            mosaic_data = {
                (r, c): props.loc[r, c]
                for r in table.index
                for c in table.columns
            }

            # Create significance labels
            label_dict = {}
            for i, r in enumerate(table.index):
                for j, c in enumerate(table.columns):
                    residual = residuals[i, j]
                    if abs(residual) > 2.58:
                        label_dict[(r, c)] = "***"
                    elif abs(residual) > 1.96:
                        label_dict[(r, c)] = "**"
                    elif abs(residual) > 1.645:
                        label_dict[(r, c)] = "*"
                    else:
                        label_dict[(r, c)] = ""

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))
            mosaic(
                mosaic_data,
                properties=lambda k: {
                    'color': cmap(norm(residuals[
                                           table.index.get_loc(k[0]),
                                           table.columns.get_loc(k[1])
                                       ]))
                },
                labelizer=lambda k: label_dict.get(k, ""),
                ax=ax
            )

            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.1)

            plt.title(f'{cond} CNV Relationships\np = {res["p_value"]:.2e}')
            plt.savefig(
                os.path.join(self.output_dir, f'mosaic_{cond}.png'),
                dpi=300, bbox_inches='tight'
            )
            plt.close()


def main():
    """Run complete analysis pipeline."""
    # Configuration
    dataset_id = "ccRCC_GBM"
    aliquot = "C3N-00495-T1_CPT0078510004"
    conditions = ["Tumor", "Normal"]  # Example conditions

    try:
        # Initialize data loader
        loader = SCENICDataLoader()
        loader.load_all_data(dataset_id, aliquot)

        # Run plotting analysis
        plotter = SCENICPlotter(loader)
        plotter.process_regulon_data()
        plotter.plot_regulon_activity_vs_cnv()
        plotter.plot_cnv_zscore_distribution()

        # Run comparative analysis
        comparator = SCENICComparer(loader)
        comparator.compare_conditions(conditions[0], conditions[1])

        print("Analysis completed successfully")

    except Exception as e:
        print(f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    main()