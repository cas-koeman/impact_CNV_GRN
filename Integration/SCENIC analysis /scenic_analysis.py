import loompy as lp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import pyreadr
from scipy.stats import kruskal, zscore, chi2_contingency
from statsmodels.graphics.mosaicplot import mosaic
from typing import Dict, List, Optional, Tuple, Union, Any


class SCENICDataLoader:
    """Handles all data loading and preprocessing for SCENIC analysis."""

    def __init__(self, base_dir: str = "/work/project/ladcol_020"):
        """Initialize with base directory path."""
        self.base_dir = base_dir
        self.loom_data: Optional[dict] = None
        self.cnv_matrix: Optional[pd.DataFrame] = None
        self.tumor_subclusters: Optional[pd.DataFrame] = None
        self.dataset_id: Optional[str] = None
        self.aliquot: Optional[str] = None

    def load_all_data(self, dataset_id: str, aliquot: str) -> None:
        """Load all required data for analysis."""
        self.dataset_id = dataset_id
        self.aliquot = aliquot
        self.load_loom_data(dataset_id, aliquot)
        self.load_cnv_matrix(dataset_id, aliquot)
        self.load_tumor_subclusters(dataset_id, aliquot)

    def load_loom_data(self, dataset_id: str, aliquot: str) -> dict:
        """Load and process SCENIC loom file."""
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

    def load_tumor_subclusters(self, dataset_id: str, aliquot: str) -> pd.DataFrame:
        """Load tumor subclusters from RDS file."""
        subcluster_path = os.path.join(
            self.base_dir, "scCNV/inferCNV",
            dataset_id, aliquot, "tumor_subclusters.rds"
        )

        if not os.path.exists(subcluster_path):
            print(f"[Warning] Tumor subclusters file not found at {subcluster_path}")
            return None

        try:
            result = pyreadr.read_r(subcluster_path)
            df = result[None]  # Extract the DataFrame from OrderedDict
            self.tumor_subclusters = df
            return df
        except Exception as e:
            print(f"[Error] Loading tumor subclusters: {str(e)}")
            return None


class CNVRegulonAnalyzer:
    """Analyzes relationships between CNV patterns and regulon activity."""

    def __init__(self, data_loader: SCENICDataLoader, sample_name: str = ""):
        """Initialize with data loader."""
        self.loader = data_loader
        self.sample_name = sample_name
        self._print_init_status()

        # Analysis results storage
        self.cnv_contingency_tables: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.cnv_association_results: Dict[str, Dict[str, dict]] = {}
        self.regulon_cnv_stats: Dict[str, Dict[str, list]] = {}

    def _print_init_status(self):
        """Print initialization status in a clean format."""
        print(f"\n[Initializing] Sample: {self.sample_name}")
        print(f"  - CNV matrix loaded: {'Yes' if self.loader.cnv_matrix is not None else 'No'}")
        print(f"  - Loom data loaded: {'Yes' if self.loader.loom_data is not None else 'No'}")
        print(f"  - Tumor subclusters loaded: {'Yes' if self.loader.tumor_subclusters is not None else 'No'}")

    def _get_cnv_status(self, gene: str, cnv_matrix: pd.DataFrame) -> str:
        """Determine CNV status for a single gene across cells."""
        if gene not in cnv_matrix.index:
            return "Neutral"

        cnv_values = cnv_matrix.loc[gene]
        gain_percent = (cnv_values > 0).mean()
        loss_percent = (cnv_values < 0).mean()

        if gain_percent >= 0.5:
            return "Gain"
        elif loss_percent >= 0.5:
            return "Loss"
        return "Neutral"

    def analyze_subcluster_populations(self) -> None:
        """Analyze CNV-regulon relationships for all tumor subclusters."""
        if self.loader.tumor_subclusters is None:
            print(f"\n[Warning] No subcluster data available for sample {self.sample_name}")
            return

        if self.loader.loom_data is None:
            raise ValueError("SCENIC loom data not loaded")
        if self.loader.cnv_matrix is None:
            raise ValueError("CNV matrix not loaded")

        # Get subcluster IDs
        subclusters = self.loader.tumor_subclusters['infercnv'].unique()
        print(f"\n[Subclusters] Distribution in {self.sample_name}:")
        print(self.loader.tumor_subclusters['infercnv'].value_counts())

        # Create cell ID conversion mapping
        cell_id_map = self._create_cell_id_mapping()
        print(f"\n[Mapping] Created cell ID map with {len(cell_id_map)} entries")

        # Initialize results storage
        self._init_results_storage()

        # Process each subcluster
        for subcluster_id in subclusters:
            self._process_subcluster(subcluster_id, cell_id_map)

    def _create_cell_id_mapping(self) -> Dict:
        """Create mapping between loom and CNV cell IDs."""
        cell_id_map = {}
        loom_cell_ids = self.loader.loom_data['cell_ids']

        for loom_id in loom_cell_ids:
            if '-' in str(loom_id):
                barcode, suffix = str(loom_id).split('-')
                cnv_style_id = f"{barcode}.{suffix}"
                if cnv_style_id in self.loader.cnv_matrix.columns:
                    cell_id_map[loom_id] = cnv_style_id
                cell_id_map[cnv_style_id] = loom_id
        return cell_id_map

    def _init_results_storage(self):
        """Initialize dictionaries to store results by subcluster."""
        self.cnv_contingency_tables[self.sample_name] = {}
        self.cnv_association_results[self.sample_name] = {}
        self.regulon_cnv_stats[self.sample_name] = {}

    def _process_subcluster(self, subcluster_id: str, cell_id_map: Dict) -> None:
        """Process a single subcluster."""
        print(f"\n[Processing] Subcluster {subcluster_id}")

        # Get cell barcodes for this subcluster
        subcluster_cells = self.loader.tumor_subclusters[
            self.loader.tumor_subclusters['infercnv'] == subcluster_id
            ]['cell'].tolist()

        # Find intersection with CNV matrix columns
        common_barcodes = self._find_common_barcodes(subcluster_cells, cell_id_map)
        if not common_barcodes:
            print(f"  [Warning] No matching cells found for subcluster {subcluster_id}")
            return

        print(f"  - Found {len(common_barcodes)} matching cells in CNV matrix")

        # Get cell indices in loom data
        subcluster_indices = self._find_loom_indices(common_barcodes, cell_id_map)
        if not subcluster_indices:
            print(f"  [Warning] No matching cells in loom data for subcluster {subcluster_id}")
            return

        print(f"  - Found {len(subcluster_indices)} matching cells in loom data")

        # Analyze the subcluster
        subcluster_name = f"Subcluster_{subcluster_id}"
        auc_matrix = self.loader.loom_data['regulons_auc'][subcluster_indices, :]
        subcluster_cnv = self.loader.cnv_matrix[list(common_barcodes)]

        self._analyze_population(auc_matrix, subcluster_indices, subcluster_name)

    def _find_common_barcodes(self, subcluster_cells: List[str], cell_id_map: Dict) -> set:
        """Find common barcodes between subcluster and CNV matrix."""
        common_barcodes = set()
        for cell in subcluster_cells:
            if cell in self.loader.cnv_matrix.columns:
                common_barcodes.add(cell)
            elif cell in cell_id_map and cell_id_map[cell] in self.loader.cnv_matrix.columns:
                common_barcodes.add(cell_id_map[cell])
        return common_barcodes

    def _find_loom_indices(self, common_barcodes: set, cell_id_map: Dict) -> list:
        """Find loom indices for cells in common barcodes."""
        loom_cell_ids = self.loader.loom_data['cell_ids']
        subcluster_indices = []

        for i, cell_id in enumerate(loom_cell_ids):
            if cell_id in common_barcodes:
                subcluster_indices.append(i)
            elif cell_id in cell_id_map and cell_id_map[cell_id] in common_barcodes:
                subcluster_indices.append(i)
            else:
                for cnv_id in common_barcodes:
                    if cnv_id in cell_id_map and cell_id_map[cnv_id] == cell_id:
                        subcluster_indices.append(i)
                        break
        return subcluster_indices

    def _analyze_population(self, auc_matrix: np.ndarray, cell_indices: list, population_name: str) -> None:
        """Analyze CNV-regulon relationships for a population."""
        print(f"\n[Analysis] Starting analysis for {population_name}")

        if self.loader.loom_data is None or self.loader.cnv_matrix is None:
            raise ValueError("Data not loaded")

        regulon_names = self.loader.loom_data['regulon_names']
        target_genes = self.loader.loom_data['target_genes']
        cnv_matrix = self.loader.cnv_matrix

        # Get cell IDs and create mapping
        cell_ids = [self.loader.loom_data['cell_ids'][i] for i in cell_indices]
        cell_id_map = self._create_cell_id_mapping_for_population(cell_ids, cnv_matrix)

        if not cell_id_map:
            print("  [Error] Cell ID mapping failed! Cannot proceed with analysis.")
            return

        # Initialize contingency tables
        full_contingency_table, simplified_table = self._init_contingency_tables()

        # Main analysis loop
        total_cell_gene_pairs = 0
        for i, regulon in enumerate(regulon_names):
            try:
                tf_name = regulon.split('_')[0].replace('(+)', '')
                if tf_name not in cnv_matrix.index:
                    continue

                target_genes_list = target_genes.get(regulon, [])
                target_genes_in_cnv = [g for g in target_genes_list if g in cnv_matrix.index]

                if not target_genes_in_cnv:
                    continue

                # Cell-level analysis
                for loom_cell_id, cnv_cell_id in cell_id_map.items():
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
                print(f"  [Warning] Skipping regulon {regulon}: {str(e)}")
                continue

        # Store and print results
        self._store_and_print_results(population_name, full_contingency_table, simplified_table, total_cell_gene_pairs)

    def _create_cell_id_mapping_for_population(self, cell_ids: List[str], cnv_matrix: pd.DataFrame) -> Dict:
        """Create cell ID mapping for a specific population."""
        cell_id_map = {}
        for loom_id in cell_ids:
            if '-' in str(loom_id):
                barcode, suffix = str(loom_id).split('-')
                cnv_style_id = f"{barcode}.{suffix}"
                if cnv_style_id in cnv_matrix.columns:
                    cell_id_map[loom_id] = cnv_style_id
            if loom_id in cnv_matrix.columns:
                cell_id_map[loom_id] = loom_id

        print(f"  - Cell ID mapping: {len(cell_id_map)}/{len(cell_ids)} cells mapped")
        return cell_id_map

    def _init_contingency_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Initialize contingency tables with small values to avoid zeros."""
        full_table = pd.DataFrame(
            1e-10,
            index=['TF Loss', 'TF Neutral', 'TF Gain'],
            columns=['Target Loss', 'Target Neutral', 'Target Gain']
        )
        simplified_table = pd.DataFrame(
            1e-10,
            index=['TF Loss', 'TF Gain'],
            columns=['Target Loss', 'Target Gain']
        )
        return full_table, simplified_table

    def _store_and_print_results(self, population_name: str, full_table: pd.DataFrame,
                                 simplified_table: pd.DataFrame, total_pairs: int) -> None:
        """Store and print analysis results."""
        # Store tables
        if self.sample_name not in self.cnv_contingency_tables:
            self.cnv_contingency_tables[self.sample_name] = {}

        self.cnv_contingency_tables[self.sample_name][population_name] = full_table
        self.cnv_contingency_tables[self.sample_name][f"{population_name}_simplified"] = simplified_table

        # Print summary
        print(f"\n[Results] {population_name}")
        print(f"  - Total TF-target gene pairs analyzed: {total_pairs:,}")
        print("\n  Full contingency table (counts):")
        print(full_table)
        print("\n  Simplified contingency table (only gains/losses):")
        print(simplified_table)

        # Statistical analysis
        self._perform_statistical_analysis(population_name, full_table, simplified_table)

    def _perform_statistical_analysis(self, population_name: str, full_table: pd.DataFrame,
                                      simplified_table: pd.DataFrame) -> None:
        """Perform and print statistical analysis results."""
        print("\n  [Statistics]")

        # 1. Full 3x3 analysis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                chi2_full, pval_full, dof_full, expected_full = chi2_contingency(
                    full_table, lambda_="log-likelihood"
                )
                n_full = full_table.values.sum()
                min_dim_full = min(full_table.shape) - 1
                cramers_v_full = np.sqrt(chi2_full / (n_full * min_dim_full))

                print(f"  Full analysis (3x3):")
                print(f"    Chi-square: {chi2_full:.3f}, p-value: {pval_full:.4e}")
                print(f"    Cramer's V: {cramers_v_full:.4f}")

                # Store full results
                self._store_analysis_results(population_name, 'full_analysis',
                                             chi2_full, pval_full, dof_full,
                                             cramers_v_full, full_table, expected_full)

            except Exception as e:
                print(f"  [Warning] Full analysis failed: {str(e)}")

        # 2. Simplified 2x2 analysis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                simplified_table_filtered = simplified_table.loc[
                    simplified_table.sum(axis=1) > 0,
                    simplified_table.sum(axis=0) > 0
                ]

                if not simplified_table_filtered.empty and simplified_table_filtered.shape == (2, 2):
                    chi2_simp, pval_simp, dof_simp, expected_simp = chi2_contingency(
                        simplified_table_filtered, lambda_="log-likelihood"
                    )
                    n_simp = simplified_table_filtered.values.sum()
                    cramers_v_simp = np.sqrt(chi2_simp / n_simp)

                    print(f"\n  Simplified analysis (2x2):")
                    print(f"    Chi-square: {chi2_simp:.3f}, p-value: {pval_simp:.4e}")
                    print(f"    Cramer's V: {cramers_v_simp:.4f}")

                    # Store simplified results
                    self._store_analysis_results(population_name, 'simplified_analysis',
                                                 chi2_simp, pval_simp, dof_simp,
                                                 cramers_v_simp, simplified_table_filtered, expected_simp)
                else:
                    print("\n  [Info] Simplified table does not have enough data for chi-square test")

            except Exception as e:
                print(f"  [Warning] Simplified analysis failed: {str(e)}")

    def _store_analysis_results(self, population_name: str, analysis_type: str,
                                chi2: float, p_value: float, dof: int,
                                cramers_v: float, table: pd.DataFrame,
                                expected: np.ndarray) -> None:
        """Store analysis results in the results dictionary."""
        if self.sample_name not in self.cnv_association_results:
            self.cnv_association_results[self.sample_name] = {}
        if population_name not in self.cnv_association_results[self.sample_name]:
            self.cnv_association_results[self.sample_name][population_name] = {}

        self.cnv_association_results[self.sample_name][population_name][analysis_type] = {
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof,
            'contingency_table': table.copy(),
            'expected': pd.DataFrame(
                expected,
                index=table.index,
                columns=table.columns
            ),
            'cramers_v': cramers_v,
            'residuals': (table.values - expected) / np.sqrt(expected)
        }

    def visualize_cnv_associations(self, population: str = None, output_dir: str = None) -> None:
        """Generate visualization of CNV-regulon relationships."""
        if not self.cnv_contingency_tables or self.sample_name not in self.cnv_contingency_tables:
            print("[Warning] No data available for visualization")
            return

        # Set default output directory
        output_dir = self._get_output_dir(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        populations_to_visualize = [population] if population else self.cnv_contingency_tables[self.sample_name].keys()

        for pop in populations_to_visualize:
            if pop.endswith("_simplified"):
                continue

            table = self.cnv_contingency_tables[self.sample_name].get(pop)
            stats = self.cnv_association_results.get(self.sample_name, {}).get(pop, {}).get('full_analysis', {})

            if table is None or not stats:
                print(f"[Warning] No data available for {pop}")
                continue

            expected = stats.get('expected')
            residuals = stats.get('residuals')

            if expected is None or residuals is None:
                print(f"[Warning] Missing data for visualization of {pop}")
                continue

            abs_max = np.max(np.abs(residuals))
            if abs_max == 0:
                print(f"[Info] No significant variation in residuals for {pop}")
                continue

            # Generate and save plot
            self._generate_mosaic_plot(pop, table, stats, residuals, abs_max, output_dir)

    def _get_output_dir(self, output_dir: Optional[str]) -> str:
        """Get the output directory path."""
        if output_dir:
            return output_dir
        return os.path.join(
            self.loader.base_dir,
            "integration_GRN_CNV",
            self.loader.dataset_id,
            self.loader.aliquot,
            "scenic_analysis"
        )

    def _generate_mosaic_plot(self, population: str, table: pd.DataFrame,
                              stats: dict, residuals: np.ndarray,
                              abs_max: float, output_dir: str) -> None:
        """Generate and save mosaic plot."""
        try:
            # Prepare plot data
            plot_data = table.div(table.sum().sum())
            mosaic_data = {
                (row, col): plot_data.loc[row, col]
                for row in table.index
                for col in table.columns
                if plot_data.loc[row, col] > 0
            }

            # Create significance labels
            label_dict = self._create_significance_labels(table, residuals)

            # Generate plot
            fig, ax = plt.subplots(figsize=(10, 8))
            mosaic(
                mosaic_data,
                properties=lambda k: {
                    'color': plt.cm.RdBu_r(
                        plt.Normalize(-abs_max, abs_max)(
                            residuals[table.index.get_loc(k[0]), table.columns.get_loc(k[1])]
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
                f"χ²({stats.get('dof', 'N/A')}) = "
                f"{stats.get('chi2', 'N/A'):.2f}, "
                f"p = {stats.get('p_value', 'N/A'):.2e}"
            )
            plt.tight_layout()

            # Save the plot
            output_path = os.path.join(output_dir, f"{population}_cnv_regulon_associations.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[Success] Saved visualization for {population} at {output_path}")

        except Exception as e:
            print(f"[Error] Failed to generate plot for {population}: {str(e)}")
            plt.close('all')

    def _create_significance_labels(self, table: pd.DataFrame, residuals: np.ndarray) -> Dict:
        """Create significance labels based on residuals."""
        label_dict = {}
        for i, row in enumerate(table.index):
            for j, col in enumerate(table.columns):
                if isinstance(residuals, np.ndarray):
                    residual = residuals[i, j]
                else:
                    residual = residuals.loc[row, col]

                if abs(residual) > 2.58:
                    label_dict[(row, col)] = "***"
                elif abs(residual) > 1.96:
                    label_dict[(row, col)] = "**"
                elif abs(residual) > 1.645:
                    label_dict[(row, col)] = "*"
        return label_dict

    def print_results_summary(self):
        """Print summary of all analysis results."""
        if not self.cnv_association_results or self.sample_name not in self.cnv_association_results:
            print("[Warning] No results to display. Run analysis first.")
            return

        print("\n" + "=" * 80)
        print(f"SUMMARY OF RESULTS FOR SAMPLE: {self.sample_name}")
        print("=" * 80)

        for population, analyses in self.cnv_association_results[self.sample_name].items():
            if population.endswith("_simplified"):
                continue

            print(f"\n[Population] {population}")

            # Full analysis results
            full_analysis = analyses.get('full_analysis', {})
            if full_analysis:
                print("\n  Full Analysis (3x3):")
                print(f"    Chi-square: {full_analysis.get('chi2', 'N/A'):.3f}")
                print(f"    p-value: {full_analysis.get('p_value', 'N/A'):.4e}")
                print(f"    Cramer's V: {full_analysis.get('cramers_v', 'N/A'):.4f}")
                print("\n    Contingency table:")
                print(full_analysis.get('contingency_table', 'N/A'))

            # Simplified analysis results
            simplified_analysis = analyses.get('simplified_analysis', {})
            if simplified_analysis:
                print("\n  Simplified Analysis (2x2 Gains/Losses only):")
                print(f"    Chi-square: {simplified_analysis.get('chi2', 'N/A'):.3f}")
                print(f"    p-value: {simplified_analysis.get('p_value', 'N/A'):.4e}")
                print(f"    Cramer's V: {simplified_analysis.get('cramers_v', 'N/A'):.4f}")

                table_key = f"{population}_simplified"
                table = self.cnv_contingency_tables.get(self.sample_name, {}).get(table_key)
                if table is not None:
                    print("\n    Simplified contingency table:")
                    print(table)

            print("\n" + "-" * 70)

    def export_results(self, dataset_id: str, aliquot: str, output_dir: Optional[str] = None) -> List[Dict]:
        """Export analysis results to CSV files."""
        output_dir = self._get_output_dir(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[Export] Saving results to {output_dir}")

        if not self.cnv_association_results or self.sample_name not in self.cnv_association_results:
            return []

        # Initialize DataFrames for results
        tumor_results = []
        subcluster_results = []
        all_results = []

        for population, analyses in self.cnv_association_results[self.sample_name].items():
            if population.endswith("_simplified"):
                continue

            # Extract data for both full and simplified analyses
            full_data = self._create_result_dict(population, analyses.get("full_analysis", {}), "full")
            simplified_data = self._create_result_dict(population, analyses.get("simplified_analysis", {}),
                                                       "simplified")

            # Append to appropriate results list
            if population == "Tumor":
                tumor_results.extend([full_data, simplified_data] if simplified_data else [full_data])
            else:
                subcluster_results.extend([full_data, simplified_data] if simplified_data else [full_data])

            all_results.extend([full_data, simplified_data] if simplified_data else [full_data])

        # Save results to files
        self._save_results_to_csv(tumor_results, "tumor_results", output_dir)
        self._save_results_to_csv(subcluster_results, "subclusters_results", output_dir)

        return all_results

    def _create_result_dict(self, population: str, analysis: dict, analysis_type: str) -> Dict:
        """Create a result dictionary for a single analysis."""
        if not analysis:
            return {}

        return {
            "sample": self.sample_name,
            "population": population,
            "analysis_type": analysis_type,
            "chi2": analysis.get("chi2"),
            "p_value": analysis.get("p_value"),
            "cramers_v": analysis.get("cramers_v"),
            "significant": analysis.get("p_value", 1.0) < 0.05,
        }

    def _save_results_to_csv(self, results: List[Dict], file_prefix: str, output_dir: str) -> None:
        """Save results to CSV file if there are any."""
        if results:
            df = pd.DataFrame(results)
            output_path = os.path.join(output_dir, f"{self.sample_name}_{file_prefix}.csv")
            df.to_csv(output_path, index=False)
            print(f"  - Saved {file_prefix} to {output_path}")


def process_sample(dataset_id: str, aliquot: str, base_dir: str = "/work/project/ladcol_020") -> Tuple[
    Optional[CNVRegulonAnalyzer], List[Dict]]:
    """Process a single sample with streamlined output."""
    print(f"\n{'=' * 80}")
    print(f"PROCESSING SAMPLE: {aliquot}")
    print(f"{'=' * 80}")

    try:
        # Load data
        loader = SCENICDataLoader(base_dir=base_dir)
        loader.load_all_data(dataset_id=dataset_id, aliquot=aliquot)

        # Create analyzer
        analyzer = CNVRegulonAnalyzer(loader, sample_name=aliquot)

        # Analyze the whole tumor population
        print("\n[Analysis] Whole tumor population")
        tumor_cell_indices = loader.loom_data['tumor_cell_indices']
        tumor_auc = loader.loom_data['regulons_auc'][tumor_cell_indices, :]
        analyzer._analyze_population(tumor_auc, tumor_cell_indices.tolist(), "Tumor")

        # Analyze tumor subclusters if available
        if loader.tumor_subclusters is not None:
            print("\n[Analysis] Tumor subclusters")
            analyzer.analyze_subcluster_populations()

        # Print summary and export results
        analyzer.print_results_summary()
        results = analyzer.export_results(dataset_id, aliquot)
        analyzer.visualize_cnv_associations()

        return analyzer, results

    except Exception as e:
        print(f"\n[Error] Processing sample {aliquot}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, []


def main():
    """Run CNV-regulon relationship analysis pipeline."""
    # Configuration
    dataset_id = "ccRCC_GBM"
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

    all_results = []
    analyzers = {}

    # Process each sample
    for sample in sample_ids:
        analyzer, sample_results = process_sample(dataset_id, sample)
        if analyzer:
            analyzers[sample] = analyzer
            all_results.extend(sample_results)

    # Save aggregate results
    if all_results:
        results_base = os.path.join(
            "/work/project/ladcol_020",
            "integration_GRN_CNV",
            dataset_id
        )
        aggregate_file = os.path.join(results_base, "allresults_scenic_analysis.csv")
        pd.DataFrame(all_results).to_csv(aggregate_file, index=False)

        # Print summary statistics
        print("\n" + "=" * 80)
        print("AGGREGATE RESULTS ACROSS ALL SAMPLES")
        print("=" * 80)
        self._print_aggregate_stats(all_results)

    print("\n[Complete] Analysis pipeline finished")


def _print_aggregate_stats(all_results: List[Dict]) -> None:
    """Print aggregate statistics from all results."""
    results_df = pd.DataFrame(all_results)

    print("\nSummary of Chi-square test significance:")
    print(results_df['significant'].value_counts())

    print("\nSummary of Cramer's V (association strength):")
    print(results_df['cramers_v'].describe())

    # Print significant results
    sig_results = results_df[results_df['significant']]
    if not sig_results.empty:
        print("\nSamples with significant associations:")
        print(sig_results[['sample', 'population', 'analysis_type', 'p_value', 'cramers_v']])
    else:
        print("\nNo samples showed significant associations")


if __name__ == "__main__":
    main()