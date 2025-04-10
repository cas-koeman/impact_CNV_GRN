import os
import warnings
from typing import Dict, Optional

import loompy as lp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from statsmodels.graphics.mosaicplot import mosaic


class SCENICDataLoader:
    """Handles all data loading and preprocessing for SCENIC analysis.

    Attributes:
        base_dir (str): Base directory path for data files
        loom_data (dict): Loaded loom file data
        cnv_matrix (pd.DataFrame): Processed CNV matrix
    """

    def __init__(self, base_dir: str = "/work/project/ladcol_020"):
        """Initialize with base directory path."""
        self.base_dir = base_dir
        self.loom_data: Optional[dict] = None
        self.cnv_matrix: Optional[pd.DataFrame] = None

    def load_all_data(self, dataset_id: str, aliquot: str) -> None:
        """Load all required data for analysis.

        Args:
            dataset_id: Dataset identifier (e.g. "ccRCC_GBM")
            aliquot: Sample aliquot ID
        """
        self.load_loom_data(dataset_id, aliquot)
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
            dataset_id, aliquot, "Tumor_pyscenic_output.loom"
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


class CNVRegulonAnalyzer:
    """Analyzes relationships between CNV patterns and regulon activity for a cell population."""

    def __init__(self, data_loader: SCENICDataLoader, sample_name: str = ""):
        """Initialize with data loader."""
        self.loader = data_loader
        self.sample_name = sample_name  # Store the sample name
        print("\nInitialized CNVRegulonAnalyzer")
        print(f"Sample: {self.sample_name}")
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
        """Analyze CNV-regulon relationships for all regulons on a per-cell basis."""
        regulon_names = self.loader.loom_data['regulon_names']
        target_genes = self.loader.loom_data['target_genes']
        cnv_matrix = self.loader.cnv_matrix

        # Get cell IDs for this population
        cell_indices = np.where(self.loader.loom_data['all_cell_types'] == population)[0]
        cell_ids = self.loader.loom_data['cell_ids'][cell_indices]

        # Handle cell ID format conversion
        cell_id_map = {}
        for loom_id in cell_ids:
            if '-' in str(loom_id):
                barcode, suffix = str(loom_id).split('-')
                cnv_style_id = f"{barcode}.{suffix}"
                if cnv_style_id in cnv_matrix.columns:
                    cell_id_map[loom_id] = cnv_style_id

        # Print mapping statistics
        print(f"\nCell ID mapping:")
        print(f"- Total cells in population '{population}': {len(cell_ids)}")
        print(f"- Successfully mapped cells: {len(cell_id_map)}")

        if len(cell_id_map) == 0:
            print("\nERROR: All cell ID mapping attempts failed! Cannot proceed with analysis.")
            return

        # Initialize full contingency table (3x3)
        full_contingency_table = pd.DataFrame(
            0,
            index=['TF Loss', 'TF Neutral', 'TF Gain'],
            columns=['Target Loss', 'Target Neutral', 'Target Gain']
        )

        # Initialize simplified 2x2 table (only gains/losses)
        simplified_table = pd.DataFrame(
            0,
            index=['TF Loss', 'TF Gain'],
            columns=['Target Loss', 'Target Gain']
        )

        # Track statistics
        total_regulons = 0
        regulons_with_tf_in_cnv = 0
        total_cells_analyzed = 0
        regulons_skipped = 0
        total_cell_gene_pairs = 0

        print(f"\nAnalyzing regulons across {len(cell_id_map)} matched cells in population '{population}'")

        # Main analysis loop
        progress_step = max(1, len(regulon_names) // 10)
        for i, regulon in enumerate(regulon_names):
            if i % progress_step == 0:
                print(f"Progress: {i}/{len(regulon_names)} regulons analyzed ({i / len(regulon_names) * 100:.1f}%)")

            total_regulons += 1
            try:
                tf_name = regulon.split('_')[0].replace('(+)', '')

                if tf_name not in cnv_matrix.index:
                    regulons_skipped += 1
                    continue

                regulons_with_tf_in_cnv += 1
                target_genes_list = target_genes.get(regulon, [])
                target_genes_in_cnv = [g for g in target_genes_list if g in cnv_matrix.index]

                if not target_genes_in_cnv:
                    continue

                # Cell-level analysis
                for loom_cell_id, cnv_cell_id in cell_id_map.items():
                    total_cells_analyzed += 1

                    # Get TF CNV status
                    tf_cnv = cnv_matrix.loc[tf_name, cnv_cell_id]
                    tf_status = 'Gain' if tf_cnv > 0 else ('Loss' if tf_cnv < 0 else 'Neutral')

                    # Get target gene CNV status
                    for target_gene in target_genes_in_cnv:
                        total_cell_gene_pairs += 1
                        target_cnv = cnv_matrix.loc[target_gene, cnv_cell_id]
                        target_status = 'Gain' if target_cnv > 0 else ('Loss' if target_cnv < 0 else 'Neutral')

                        # Update both tables
                        full_contingency_table.loc[f"TF {tf_status}", f"Target {target_status}"] += 1

                        # Only update simplified table for non-neutral cases
                        if tf_status != 'Neutral' and target_status != 'Neutral':
                            simplified_table.loc[f"TF {tf_status}", f"Target {target_status}"] += 1

            except Exception as e:
                print(f"Skipping regulon {regulon} due to error: {str(e)}")
                continue

        # Print summary statistics
        print(f"\nAnalysis complete!")
        print(f"Total regulons analyzed: {total_regulons}")
        print(f"Regulons with TF in CNV file: {regulons_with_tf_in_cnv}")
        print(f"Regulons skipped (TF not in CNV): {regulons_skipped}")
        print(f"Total cells analyzed: {total_cells_analyzed}")
        print(f"Total TF-target gene pairs analyzed: {total_cell_gene_pairs}")

        # Store tables
        self.cnv_contingency_tables[population] = full_contingency_table
        self.cnv_contingency_tables[f"{population}_simplified"] = simplified_table

        # Print both tables
        print("\nFull contingency table (counts):")
        print(full_contingency_table)
        print("\nSimplified contingency table (only gains/losses):")
        print(simplified_table)

        # Statistical analysis
        if not full_contingency_table.empty:
            print("\n=== STATISTICAL ANALYSIS ===")

            # 1. Full 3x3 analysis
            print("\n1. FULL ANALYSIS (3x3):")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    chi2_full, pval_full, dof_full, expected_full = chi2_contingency(
                        full_contingency_table,
                        lambda_="log-likelihood"
                    )
                    n_full = full_contingency_table.values.sum()
                    min_dim_full = min(full_contingency_table.shape) - 1
                    cramers_v_full = np.sqrt(chi2_full / (n_full * min_dim_full))

                    print(f"Chi-square statistic: {chi2_full:.4f}")
                    print(f"P-value: {pval_full:.4e}")
                    print(f"Degrees of freedom: {dof_full}")
                    print(f"Cramer's V: {cramers_v_full:.4f}")

                    # Store full results
                    self.cnv_association_results[population] = {
                        'full_analysis': {
                            'chi2': chi2_full,
                            'p_value': pval_full,
                            'dof': dof_full,
                            'expected': pd.DataFrame(
                                expected_full,
                                index=full_contingency_table.index,
                                columns=full_contingency_table.columns
                            ),
                            'cramers_v': cramers_v_full,
                            'residuals': (full_contingency_table.values - expected_full) / np.sqrt(expected_full)
                        }
                    }

                except Exception as e:
                    print(f"Full analysis chi-square test failed: {str(e)}")

            # 2. Simplified 2x2 analysis
            print("\n2. SIMPLIFIED ANALYSIS (2x2 Gains/Losses only):")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    # Filter out empty rows/columns in case some categories are missing
                    simplified_table = simplified_table.loc[
                        simplified_table.sum(axis=1) > 0,
                        simplified_table.sum(axis=0) > 0
                    ]

                    if not simplified_table.empty:
                        chi2_simp, pval_simp, dof_simp, expected_simp = chi2_contingency(
                            simplified_table,
                            lambda_="log-likelihood"
                        )
                        n_simp = simplified_table.values.sum()
                        cramers_v_simp = np.sqrt(chi2_simp / n_simp)  # For 2x2, min_dim is always 1

                        print(f"Chi-square statistic: {chi2_simp:.4f}")
                        print(f"P-value: {pval_simp:.4e}")
                        print(f"Degrees of freedom: {dof_simp}")
                        print(f"Cramer's V: {cramers_v_simp:.4f}")

                        # Store simplified results
                        self.cnv_association_results[population]['simplified_analysis'] = {
                            'chi2': chi2_simp,
                            'p_value': pval_simp,
                            'dof': dof_simp,
                            'expected': pd.DataFrame(
                                expected_simp,
                                index=simplified_table.index,
                                columns=simplified_table.columns
                            ),
                            'cramers_v': cramers_v_simp,
                            'residuals': (simplified_table.values - expected_simp) / np.sqrt(expected_simp)
                        }

                except Exception as e:
                    print(f"Simplified analysis chi-square test failed: {str(e)}")

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
            # fig = analyzer.plot_cnv_sankey(cell_population)
            # Save with sample name in filename
            # fig.write_html(f"{sample}_tf_cnv_to_target_relationships.html")  # Interactive version

        except Exception as e:
            print(f"Analysis failed for sample {sample}: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
