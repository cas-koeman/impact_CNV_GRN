import loompy as lp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy.stats import kruskal, zscore, chi2_contingency
from statsmodels.graphics.mosaicplot import mosaic
from typing import Dict, Tuple, Optional, List
import plotly.graph_objects as go


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


class CNVRegulonAnalyzer:
    """Analyzes relationships between CNV patterns and regulon activity for a cell population."""

    def __init__(self, data_loader: SCENICDataLoader, sample_name: str = ""):
        """Initialize with data loader."""
        self.loader = data_loader
        self.sample_name = sample_name  # Store the sample name
        print("\nInitialized CNVRegulonAnalyzer")
        print(f"Sample: {self.sample_name}")
        print(f"Expression matrix loaded: {self.loader.expression_matrix is not None}")
        print(f"CNV matrix loaded: {self.loader.cnv_matrix is not None}")
        print(f"Loom data loaded: {self.loader.loom_data is not None}\n")

        # Analysis results storage
        self.cnv_contingency_tables: Dict[str, pd.DataFrame] = {}
        self.cnv_association_results: Dict[str, dict] = {}
        self.regulon_cnv_stats: Dict[str, dict] = {}

    def _get_cnv_status(self, gene: str, cnv_matrix: pd.DataFrame) -> str:
        """
        Determine CNV status for a single gene across cells.

        Args:
            gene: Gene name to check
            cnv_matrix: DataFrame of CNV values

        Returns:
            'Gain', 'Loss', or 'Neutral' based on CNV status
        """
        if gene not in cnv_matrix.index:
            return "Neutral"  # Gene not in CNV matrix

        cnv_values = cnv_matrix.loc[gene]

        # Calculate percentage of cells with gains/losses
        gain_percent = (cnv_values > 0).mean()
        loss_percent = (cnv_values < 0).mean()

        if gain_percent >= 0.5:
            return "Gain"
        elif loss_percent >= 0.5:
            return "Loss"
        else:
            return "Neutral"

    def analyze_cell_population(self, cell_population: str) -> None:
        """Analyze CNV-regulon relationships for all regulons in a cell population."""
        print(f"\n{'=' * 50}")
        print(f"Starting analysis for sample: {self.sample_name}")
        print(f"Cell population: {cell_population}")
        print(f"{'=' * 50}")

        if self.loader.loom_data is None:
            raise ValueError("SCENIC loom data not loaded")
        if self.loader.cnv_matrix is None:
            raise ValueError("CNV matrix not loaded")

        loom_data = self.loader.loom_data
        cell_types = loom_data['all_cell_types']

        print("\nCell type distribution:")
        print(pd.Series(cell_types).value_counts())

        pop_idx = np.where(cell_types == cell_population)[0]
        print(f"\nFound {len(pop_idx)} cells in population '{cell_population}'")

        if len(pop_idx) == 0:
            raise ValueError(f"Cell population '{cell_population}' not found")

        print("\nExtracting AUC matrix...")
        auc_matrix = loom_data['regulons_auc'][pop_idx, :]
        print(f"AUC matrix shape: {auc_matrix.shape} (cells x regulons)")
        print(f"First 5 regulons: {loom_data['regulon_names'][:5]}")

        self._analyze_all_regulons(auc_matrix, cell_population)

    def _analyze_all_regulons(self, auc_matrix: np.ndarray, population: str) -> None:
        regulon_names = self.loader.loom_data['regulon_names']
        target_genes = self.loader.loom_data['target_genes']
        cnv_matrix = self.loader.cnv_matrix

        # Initialize contingency table with zeros
        contingency_table = pd.DataFrame(
            0,
            index=['TF Loss', 'TF Neutral', 'TF Gain'],
            columns=['Target Loss', 'Target Neutral', 'Target Gain']
        )

        # Dictionary to store TF CNV status counts (only for TFs in CNV file)
        tf_cnv_counts = {'Loss': 0, 'Neutral': 0, 'Gain': 0}
        tf_status_details = {}  # Store details per TF
        regulon_stats = []

        # Counters for tracking regulons and target genes
        total_regulons = 0
        regulons_with_tf_in_cnv = 0
        regulons_skipped = 0
        total_target_genes = 0  # New counter for target genes

        for regulon in regulon_names:
            total_regulons += 1
            try:
                # Extract TF name from regulon (format: TF_MOTIF) and remove (+) suffix
                tf_name = regulon.split('_')[0].replace('(+)', '')

                # Skip this regulon if TF is not in CNV matrix
                if tf_name not in cnv_matrix.index:
                    regulons_skipped += 1
                    continue

                regulons_with_tf_in_cnv += 1

                # Get CNV status for TF
                tf_status = self._get_cnv_status(tf_name, cnv_matrix)

                # Update TF CNV counts and store details
                tf_cnv_counts[tf_status] += 1
                tf_status_details[tf_name] = tf_status

                # Calculate CNV status for target genes
                target_status_counts = {'Gain': 0, 'Loss': 0, 'Neutral': 0}
                target_genes_list = target_genes.get(regulon, [])
                total_target_genes += len(target_genes_list)  # Add to target gene counter

                for gene in target_genes_list:
                    # Get CNV status for this target gene
                    target_status = self._get_cnv_status(gene, cnv_matrix)
                    target_status_counts[target_status] += 1

                    # Update contingency table for each target gene
                    row_name = f"TF {tf_status}"
                    col_name = f"Target {target_status}"
                    contingency_table.loc[row_name, col_name] += 1

                # Store regulon statistics
                total_targets = sum(target_status_counts.values())
                if total_targets > 0:
                    target_percentages = {
                        k: v / total_targets for k, v in target_status_counts.items()
                    }
                else:
                    target_percentages = {'Gain': 0, 'Loss': 0, 'Neutral': 1}

                regulon_stats.append({
                    'regulon': regulon,
                    'tf': tf_name,
                    'tf_status': tf_status,
                    'target_counts': target_status_counts,
                    'target_percentages': target_percentages,
                    'total_targets': total_targets
                })

            except Exception as e:
                print(f"Skipping regulon {regulon} due to error: {str(e)}")
                continue

        # Calculate average number of target genes per regulon (only for regulons with TF in CNV)
        avg_target_genes = total_target_genes / regulons_with_tf_in_cnv if regulons_with_tf_in_cnv > 0 else 0

        # Print summary statistics
        print(f"\nTotal regulons analyzed: {total_regulons}")
        print(f"Regulons with TF in CNV file: {regulons_with_tf_in_cnv}")
        print(f"Regulons skipped (TF not in CNV): {regulons_skipped}")
        print(f"\nAverage number of target genes per regulon (for regulons with TF in CNV): {avg_target_genes:.1f}")

        # Print TF CNV status summary (only for TFs in CNV file)
        print("\nTranscription Factor CNV Status Summary (only TFs in CNV file):")
        print(f"TFs with Loss: {tf_cnv_counts['Loss']}")
        print(f"TFs with Neutral: {tf_cnv_counts['Neutral']}")
        print(f"TFs with Gain: {tf_cnv_counts['Gain']}")

        # Store results
        self.regulon_cnv_stats[population] = regulon_stats
        self.tf_cnv_status = {
            'counts': tf_cnv_counts,
            'details': tf_status_details
        }

        print("\nFinal contingency table (counts per target gene):")
        print(contingency_table)

        # Filter out empty rows/columns
        contingency_table = contingency_table.loc[
            contingency_table.sum(axis=1) > 0,
            contingency_table.sum(axis=0) > 0
        ]

        if contingency_table.empty:
            print(f"\nWarning: No CNV-regulon relationships found for {population}")
            return

        self.cnv_contingency_tables[population] = contingency_table

        # Perform chi-square test if we have data
        if not contingency_table.empty:
            print("\nPerforming chi-square test...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    chi2, pval, dof, expected = chi2_contingency(
                        contingency_table,
                        lambda_="log-likelihood"
                    )
                    print(f"Chi2 statistic: {chi2:.3f}")
                    print(f"P-value: {pval:.4e}")
                    print(f"Degrees of freedom: {dof}")

                    self.cnv_association_results[population] = {
                        'chi2_statistic': chi2,
                        'p_value': pval,
                        'degrees_of_freedom': dof,
                        'expected_frequencies': expected,
                        'contingency_table': contingency_table
                    }

                    print("\nExpected frequencies:")
                    print(pd.DataFrame(
                        expected,
                        index=contingency_table.index,
                        columns=contingency_table.columns
                    ))

                except Exception as e:
                    print(f"Statistical test failed: {str(e)}")

            # Visualize results
            self._visualize_cnv_associations(population)

    def _visualize_cnv_associations(self, population: str) -> None:
        """Generate visualization of CNV-regulon relationships."""
        print(f"\nGenerating visualization for {population}...")

        table = self.cnv_contingency_tables.get(population)
        if table is None:
            print("No data available for visualization")
            return

        stats = self.cnv_association_results.get(population, {})
        expected = stats.get('expected_frequencies')

        if expected is None:
            print("No expected frequencies available")
            return

        # Calculate standardized residuals
        with np.errstate(divide='ignore', invalid='ignore'):
            residuals = np.nan_to_num(
                (table.values - expected) / np.sqrt(expected),
                nan=0, posinf=0, neginf=0
            )

        print("\nStandardized residuals:")
        print(pd.DataFrame(
            residuals,
            index=table.index,
            columns=table.columns
        ))

        abs_max = np.max(np.abs(residuals))
        if abs_max == 0:
            print("No significant variation in residuals")
            return

        # Prepare plot data
        plot_data = table.div(table.sum().sum())
        mosaic_data = {
            (row, col): plot_data.loc[row, col]
            for row in table.index
            for col in table.columns
        }

        # Create significance labels
        label_dict = {}
        for i, row in enumerate(table.index):
            for j, col in enumerate(table.columns):
                residual = residuals[i, j]
                if abs(residual) > 2.58:
                    label_dict[(row, col)] = "***"
                elif abs(residual) > 1.96:
                    label_dict[(row, col)] = "**"
                elif abs(residual) > 1.645:
                    label_dict[(row, col)] = "*"

        # Generate plot
        print("\nCreating mosaic plot...")
        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create mosaic plot
            mosaic(
                mosaic_data,
                properties=lambda k: {
                    'color': plt.cm.RdBu_r(
                        plt.Normalize(-abs_max, abs_max)(
                            residuals[
                                table.index.get_loc(k[0]),
                                table.columns.get_loc(k[1])
                            ]
                        )
                    )
                },
                labelizer=lambda k: label_dict.get(k, ""),
                ax=ax,
                label_rotation=45
            )

            # Add colorbar
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.RdBu_r,
                norm=plt.Normalize(-abs_max, abs_max))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="Standardized Residual")

            plt.title(
                f"{population} CNV-Regulon Associations\n"
                f"χ²({stats.get('degrees_of_freedom', 'N/A')}) = "
                f"{stats.get('chi2_statistic', 'N/A'):.2f}, "
                f"p = {stats.get('p_value', 'N/A'):.2e}"
            )
            plt.tight_layout()
            plt.savefig(
                f"{population}_cnv_regulon_associations.png",
                dpi=300,
                bbox_inches='tight'
            )
            print(f"Saved plot to {population}_cnv_regulon_associations.png")
            plt.close()
        except Exception as e:
            print(f"Failed to generate plot: {str(e)}")
            plt.close('all')

    def plot_cnv_sankey(self, population: str):
        """Generate Sankey plot with Gain(Red)-Neutral(Gray)-Loss(Blue) color scheme
        Only includes transcription factors found in the CNV file"""
        if population not in self.regulon_cnv_stats:
            raise ValueError(f"No data available for population {population}")

        # Get the pre-computed regulon stats and sort by TF CNV status (Gain > Neutral > Loss)
        regulon_stats = sorted(
            self.regulon_cnv_stats[population],
            key=lambda x: {'Gain': 0, 'Neutral': 1, 'Loss': 2}[x['tf_status']]
        )

        # Print number of TFs found in CNV file
        tf_in_cnv_count = len({stat['tf'] for stat in regulon_stats})
        print(f"\nNumber of transcription factors found in CNV file: {tf_in_cnv_count}")

        # Define consistent order (Gain -> Neutral -> Loss)
        cnv_order = ['Gain', 'Neutral', 'Loss']

        # Define colors for each CNV status
        cnv_colors = {
            'Gain': '#FF6B6B',  # Red for Gain
            'Neutral': '#D3D3D3',  # Gray for Neutral
            'Loss': '#A6CEE3'  # Blue for Loss
        }

        # Initialize data structures
        links = {'source': [], 'target': [], 'value': []}

        # Build node list with ordered TFs first
        node_labels = []
        node_colors = []

        # 1. Left column: TFs colored by their CNV status (only those in CNV file)
        tf_to_index = {}
        for i, stat in enumerate(regulon_stats):
            tf_name = stat['tf']
            if tf_name not in tf_to_index:
                tf_to_index[tf_name] = len(node_labels)
                node_labels.append(tf_name)
                # Color TFs based on their CNV status
                node_colors.append(cnv_colors[stat['tf_status']])

        # 2. Middle column: TF CNV status (Red->Gray->Blue)
        status_start_idx = len(node_labels)
        node_labels.extend([f'TF {status}' for status in cnv_order])
        node_colors.extend([cnv_colors[status] for status in cnv_order])

        # 3. Right column: Target CNV status (Red->Gray->Blue)
        target_start_idx = len(node_labels)
        node_labels.extend([f'Target {status}' for status in cnv_order])
        node_colors.extend([cnv_colors[status] for status in cnv_order])

        # Build links
        for stat in regulon_stats:
            tf_idx = tf_to_index[stat['tf']]
            tf_status = stat['tf_status']

            # Link TF to its status
            status_idx = status_start_idx + cnv_order.index(tf_status)
            links['source'].append(tf_idx)
            links['target'].append(status_idx)
            links['value'].append(stat['total_targets'])

            # Link TF status to target statuses
            for status in cnv_order:
                count = stat['target_counts'].get(status, 0)
                if count > 0:
                    target_idx = target_start_idx + cnv_order.index(status)
                    links['source'].append(status_idx)
                    links['target'].append(target_idx)
                    links['value'].append(count)

        # Create the Sankey diagram
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=5,
                thickness=25,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors,
                x=[0] * len(tf_to_index) + [0.3] * 3 + [0.7] * 3,  # Column positions
                y=[i / len(tf_to_index) for i in range(len(tf_to_index))] +
                  [0.1, 0.5, 0.9] * 2  # More spread out vertically
            ),
            link=dict(
                source=links['source'],
                target=links['target'],
                value=links['value'],
                color=[node_colors[links['source'][i]] for i in range(len(links['source']))]  # Color links by source
            )
        ))

        # Add minimal column headers
        fig.add_annotation(x=0.0, y=1.05, text="Regulons", showarrow=False, font_size=12)
        fig.add_annotation(x=0.3, y=1.05, text="TF CNV", showarrow=False, font_size=12)
        fig.add_annotation(x=0.7, y=1.05, text="Target CNV", showarrow=False, font_size=12)

        return fig

def main():
    """Run CNV-regulon relationship analysis pipeline."""
    # Configuration
    dataset_id = "ccRCC_GBM"
    cell_population = "Tumor"  # The cell population to analyze

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

            # Run CNV-regulon analysis
            print(f"Analyzing CNV-regulon relationships for {cell_population} cells in sample {sample}...")
            analyzer = CNVRegulonAnalyzer(loader, sample)
            analyzer.analyze_cell_population(cell_population)

            print(f"Analysis completed successfully for sample {sample}")

            # Generate the Sankey plot
            fig = analyzer.plot_cnv_sankey(cell_population)
            # Save with sample name in filename
            fig.write_html(f"{sample}_tf_cnv_to_target_relationships.html")  # Interactive version

        except Exception as e:
            print(f"Analysis failed for sample {sample}: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()