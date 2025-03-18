import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3, venn2
from scipy.stats import zscore, chi2_contingency
import os
from matplotlib import colors as mcolors
from statsmodels.graphics.mosaicplot import mosaic

class RegfileComparison:
    def __init__(self, dataset_id, aliquot, conditions, base_dir="/work/project/ladcol_020"):
        self.dataset_id = dataset_id
        self.aliquot = aliquot
        self.conditions = conditions
        self.base_dir = base_dir

        # Auto-generate file paths
        scenic_dir = os.path.join(base_dir, "scGRNi/RNA/SCENIC", dataset_id, aliquot)
        self.file_paths = [
            os.path.join(scenic_dir, f"reg.csv") if condition == "all" else os.path.join(scenic_dir, f"{condition}_reg.csv")
            for condition in conditions
        ]

        self.output_dir = os.path.join(dataset_id, aliquot)

        # Data containers
        self.dfs = [None] * len(conditions)
        self.expression_matrix = None
        self.cnv_matrix = None
        self.expression_data_combined = None
        self.cnv_data_combined = None
        self.contingency_tables = {}
        self.chi2_results = {}

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        print("Loading data...")
        for i, (condition, file_path) in enumerate(zip(self.conditions, self.file_paths)):
            if os.path.exists(file_path):
                self.dfs[i] = pd.read_csv(file_path, header=1).iloc[1:].reset_index(drop=True)
                self.dfs[i] = self.dfs[i].rename(columns={'Unnamed: 0': 'TF', 'Unnamed: 1': 'MotifID'})
                print(f"Loaded {condition} with {len(self.dfs[i])} rows from {file_path}")
            else:
                print(f"Warning: File not found for {condition} at {file_path}")

        # Load gene expression data and apply filters
        expression_file = os.path.join(
            self.base_dir,
            "integration_visualization",
            self.dataset_id,
            self.aliquot,
            "raw_count_matrix.txt"
        )

        print(f"Loading and filtering gene expression data from {expression_file}...")
        if os.path.exists(expression_file):
            self.expression_matrix = pd.read_csv(expression_file, sep='\t', index_col=0)
        else:
            print(f"Warning: Expression file not found at {expression_file}")

        # Filter genes expressed in at least 200 cells
        gene_filter = (self.expression_matrix > 0).sum(axis=1) >= 200
        filtered_expression_matrix = self.expression_matrix[gene_filter]

        # Filter cells with at least 3 expressed genes
        cell_filter = (filtered_expression_matrix > 0).sum(axis=0) >= 3
        self.expression_matrix = filtered_expression_matrix.loc[:, cell_filter]

        print(
            f"Filtered gene expression data to {self.expression_matrix.shape[0]} genes and {self.expression_matrix.shape[1]} cells.")

        # Load CNV data
        print("Loading CNV data...")
        self.cnv_matrix = pd.read_csv(
            f"/work/project/ladcol_020/integration_visualization/{self.dataset_id}/{self.aliquot}/cnv_matrix.tsv",
            sep='\t', index_col=0
        )
        self.cnv_matrix = self.cnv_matrix * 2 - 2

        print(f"Loaded CNV data with {self.cnv_matrix.shape[0]} genes and {self.cnv_matrix.shape[1]} cells.")

    def prepare_data(self):
        print("Preparing data...")
        for i, df in enumerate(self.dfs):
            if df is not None:
                df['TargetGeneCount'] = df['TargetGenes'].apply(self._count_target_genes)
                df['Condition'] = self.conditions[i]

        print("Data preparation complete.")

    @staticmethod
    def _count_target_genes(target_genes_str):
        if pd.isna(target_genes_str):
            return 0
        try:
            return len(eval(target_genes_str))
        except:
            return 0

    @staticmethod
    def _extract_gene_names(target_genes_str):
        if pd.isna(target_genes_str):
            return []
        try:
            target_genes_list = eval(target_genes_str)
            return [gene[0] for gene in target_genes_list]
        except:
            return []

    @staticmethod
    def _classify_cnv(cnv_value):
        if cnv_value < 0:
            return 'Loss'
        elif cnv_value == 0:
            return 'Neutral'
        else:
            return 'Gain'

    def _get_cnv_status(self, genes):
        gene_cnv = self.cnv_matrix.loc[self.cnv_matrix.index.intersection(genes)]
        avg_cnv = gene_cnv.mean(axis=1)
        return avg_cnv.apply(self._classify_cnv)

    def plot_venn_diagram(self):
        if len(self.conditions) < 2:
            print("Not enough conditions to plot Venn diagram.")
            return

        print("Plotting Venn diagram...")
        sets = [set(df['TF']) for df in self.dfs if df is not None]
        set_labels = [f'{condition} Regulons' for condition in self.conditions]

        plt.figure(figsize=(10, 8))
        if len(sets) == 2:
            venn = venn2(sets, set_labels=set_labels)
        elif len(sets) == 3:
            venn = venn3(sets, set_labels=set_labels)
        else:
            print("Venn diagram not supported for more than 3 conditions.")
            return

        plt.title(f'Overlap of {", ".join(self.conditions)} SCENIC Derived Regulons', fontsize=12)
        plt.savefig(os.path.join(self.output_dir, 'tf_overlap_venn.png'), dpi=300)
        plt.close()

        print("Venn diagram saved as 'tf_overlap_venn.png'.")

    def prepare_expression_data(self):
        print("Preparing expression data...")
        target_genes = [df['TargetGenes'].apply(self._extract_gene_names).explode().dropna().unique() for df in self.dfs if df is not None]
        expressions = [self.expression_matrix.loc[self.expression_matrix.index.intersection(genes)] for genes in target_genes]
        expressions_zscore = [expression.apply(zscore, axis=1) for expression in expressions]
        expressions_flat = [expression_zscore.values.flatten() for expression_zscore in expressions_zscore]

        self.expression_data_combined = pd.DataFrame({
            'Condition': [condition for condition, flat in zip(self.conditions, expressions_flat) for _ in range(len(flat))],
            'Expression': [item for sublist in expressions_flat for item in sublist]
        })

        print(f"Prepared expression data for {len(self.expression_data_combined)} genes.")

    def prepare_cnv_data(self):
        print("Preparing CNV data...")
        target_genes = [df['TargetGenes'].apply(self._extract_gene_names).explode().dropna().unique() for df in self.dfs if df is not None]
        cnvs = [self.cnv_matrix.loc[self.cnv_matrix.index.intersection(genes)] for genes in target_genes]
        cnvs_flat = [cnv.values.flatten() for cnv in cnvs]

        self.cnv_data_combined = pd.DataFrame({
            'Condition': [condition for condition, flat in zip(self.conditions, cnvs_flat) for _ in range(len(flat))],
            'CNV': [item for sublist in cnvs_flat for item in sublist]
        })

        print(f"Prepared CNV data for {len(self.cnv_data_combined)} genes.")

    def plot_violin_and_stacked_bar(self):
        print("Plotting violin and stacked bar plots...")
        violin_data = pd.DataFrame({
            'Condition': [condition for condition, df in zip(self.conditions, self.dfs) for _ in range(len(df))],
            'Target Gene Count': [item for df in self.dfs for item in df['TargetGeneCount']]
        })

        fig, axes = plt.subplots(1, 3, figsize=(12, 6))

        # Subplot 1: Target Gene Count Violin Plot
        sns.violinplot(x='Condition', y='Target Gene Count', data=violin_data, palette='viridis', ax=axes[0],
                       inner=None, gridsize=1000)
        axes[0].set_ylabel('Number of Target Genes within Regulons')

        # Subplot 2: Expression Violin Plot (Z-score normalized)
        sns.violinplot(x='Condition', y='Expression', data=self.expression_data_combined, palette='viridis', ax=axes[1],
                       inner=None, gridsize=3000)
        axes[1].set_ylabel('Z-normalized Gene Expression')
        axes[1].set_ylim(-2, 3)

        # Subplot 3: Stacked Bar Plot for CNV Proportions
        cnv_counts = self.cnv_data_combined.groupby(['Condition', 'CNV']).size().reset_index(name='Count')
        total_counts = cnv_counts.groupby('Condition')['Count'].sum().reset_index(name='Total')
        cnv_counts = cnv_counts.merge(total_counts, on='Condition')
        cnv_counts['Proportion'] = cnv_counts['Count'] / cnv_counts['Total']
        cnv_pivot = cnv_counts.pivot(index='Condition', columns='CNV', values='Proportion').fillna(0)

        cnv_values = cnv_pivot.columns
        viridis_colors = sns.color_palette('viridis', n_colors=len(cnv_values))
        cnv_pivot.plot(kind='bar', stacked=True, color=viridis_colors, edgecolor='black', linewidth=1.5, ax=axes[2])
        axes[2].set_xlabel('Copy Number')
        axes[2].set_ylabel('Proportion of Target Genes per Copy Number')
        axes[2].set_ylim(0, 1)
        axes[2].legend(title='Copy Number Variation', fontsize=10, bbox_to_anchor=(1.03, 0.5), loc='center left')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'regulon_comparison.png'), dpi=300)
        plt.close()

        print("Violin and stacked bar plots saved as 'regulon_comparison.png'.")

    def perform_statistical_tests(self):
        print("Performing statistical tests...")
        for condition, df in zip(self.conditions, self.dfs):
            if df is not None:
                self.contingency_tables[condition] = self._create_tf_target_contingency_table(df)
                chi2, p_value, dof, expected = chi2_contingency(self.contingency_tables[condition])
                self.chi2_results[condition] = {
                    'chi2': chi2,
                    'p_value': p_value,
                    'dof': dof,
                    'expected': expected
                }

        self.plot_mosaic_cnv()

    def _create_tf_target_contingency_table(self, df):
        print("Creating TF-target contingency table...")
        cnv_categories = ['Loss', 'Neutral', 'Gain']
        contingency_table = pd.DataFrame(0, index=cnv_categories, columns=[f'TG {c}' for c in cnv_categories])

        empty_count = 0

        for _, row in df.iterrows():
            tf = row['TF']
            target_genes = self._extract_gene_names(row['TargetGenes'])

            if not target_genes:
                empty_count += 1
                print(f"No target genes found for TF {tf}. Skipping...")
                continue

            tf_cnv_status = self._get_cnv_status([tf]).get(tf, 'Neutral')
            target_cnv_status = self._get_cnv_status(target_genes)

            for _, status in target_cnv_status.items():
                contingency_table.loc[tf_cnv_status, f'TG {status}'] += 1

        print(f"Total number of entries with no target genes: {empty_count}")

        return contingency_table

    def plot_mosaic_cnv(self):
        print("Creating mosaic plots with Pearson residual-based coloring...")
        cmap = plt.cm.RdBu_r

        for condition, contingency_table in self.contingency_tables.items():
            chi2_result = self.chi2_results[condition]
            expected = chi2_result['expected']

            observed = contingency_table.values
            residuals = (observed - expected) / np.sqrt(expected)

            abs_max_residual = np.max(np.abs(residuals))
            norm = plt.Normalize(-abs_max_residual, abs_max_residual)

            row_labels = ['TF Loss', 'TF Neutral', 'TF Gain']
            col_labels = ['TG Loss', 'TG Neutral', 'TG Gain']

            contingency_table.index = row_labels
            contingency_table.columns = col_labels

            proportions = contingency_table.div(contingency_table.sum(axis=1), axis=0)

            mosaic_data = {}
            labelizer_dict = {}

            for i, row_label in enumerate(row_labels):
                for j, col_label in enumerate(col_labels):
                    mosaic_data[(row_label, col_label)] = proportions.iloc[i, j]

                    residual = residuals[i, j]
                    if abs(residual) > 2.58:
                        labelizer_dict[(row_label, col_label)] = "***"
                    elif abs(residual) > 1.96:
                        labelizer_dict[(row_label, col_label)] = "**"
                    elif abs(residual) > 1.645:
                        labelizer_dict[(row_label, col_label)] = "*"
                    else:
                        labelizer_dict[(row_label, col_label)] = ""

            def props(key):
                residual = residuals[row_labels.index(key[0]), col_labels.index(key[1])]
                return {"color": cmap(norm(residual))}

            labelizer_func = lambda key: f"{labelizer_dict.get(key, '')}"

            fig, ax = plt.subplots(figsize=(8, 6))
            mosaic(mosaic_data, title=f'TF-Target Gene CNV Relationships - {condition}',
                   properties=props, labelizer=labelizer_func, ax=ax)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.08)
            cbar.set_label('Pearson Residual')

            p_value = chi2_result['p_value']
            plt.figtext(0.9, 0.08, f"p = {p_value:.3e}", ha='center', fontsize=12)

            filename = f"{condition.lower()}_cnv_mosaic.png"
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            plt.close()

            print(f"Saved mosaic plot for {condition} as {filename}")

    def analyze(self):
        print("Starting analysis...\n")
        self.load_data()
        self.prepare_data()
        self.prepare_expression_data()
        self.prepare_cnv_data()
        self.plot_venn_diagram()
        self.perform_statistical_tests()

        print("\nContingency Tables\n" + "-" * 40)
        for condition, table in self.contingency_tables.items():
            print(f"\nDataset: {condition}")
            print(table, "\n")

        print("\nChi-Square Test Results\n" + "-" * 40)
        for label, result in self.chi2_results.items():
            print(f"Dataset: {label}")
            print(f"   - Chi2 Value: {result['chi2']:.3f}")
            print(f"   - p-value: {result['p_value']:.5f}")
            print(f"   - Degrees of Freedom: {result['dof']}")
            print("   - Expected Frequencies:\n", pd.DataFrame(result['expected']), "\n")


# Example usage
if __name__ == "__main__":
    # List of aliquots to process
    aliquots = [
        "C3L-00004-T1_CPT0001540013",
        "C3N-00495-T1_CPT0078510004",
        "C3L-00917-T1_CPT0023690004",
        "C3L-00088-T1_CPT0000870003",
        "C3L-00026-T1_CPT0001500003",
        "C3L-00448-T1_CPT0010160004",
        "C3L-01313-T1_CPT0086820004",
        "C3L-00416-T2_CPT0010100001"
    ]

    dataset_id = "ccRCC_GBM"

    # Process each aliquot
    for aliquot in aliquots:
        print(f"\nProcessing aliquot: {aliquot}")
        conditions = ["all", "Tumor"]  # Example conditions
        comparison = RegfileComparison(dataset_id, aliquot, conditions)
        comparison.analyze()