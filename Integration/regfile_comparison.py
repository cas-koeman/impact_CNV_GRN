import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3
from scipy.stats import zscore, chi2_contingency
import matplotlib.patches as patches
import os
from statsmodels.graphics.mosaicplot import mosaic


class RegfileComparison:
    def __init__(self, file_path1, condition1, dataset_id, aliquot, file_path2=None, file_path3=None, condition2=None,
                 condition3=None):
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.file_path3 = file_path3
        self.condition1 = condition1
        self.condition2 = condition2
        self.condition3 = condition3
        self.dataset_id = dataset_id
        self.aliquot = aliquot
        self.output_dir = os.path.join(dataset_id, aliquot)

        # Data containers
        self.df1 = None
        self.df2 = None
        self.df3 = None
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

        # Load dataset 1 (always required)
        self.df1 = pd.read_csv(self.file_path1, header=1).iloc[1:].reset_index(drop=True)
        self.df1 = self.df1.rename(columns={'Unnamed: 0': 'TF', 'Unnamed: 1': 'MotifID'})
        print(f"Loaded {self.condition1} with {len(self.df1)} rows.")

        # Load dataset 2 if provided
        if self.file_path2:
            self.df2 = pd.read_csv(self.file_path2, header=1).iloc[1:].reset_index(drop=True)
            self.df2 = self.df2.rename(columns={'Unnamed: 0': 'TF', 'Unnamed: 1': 'MotifID'})
            print(f"Loaded {self.condition2} with {len(self.df2)} rows.")
        else:
            print(f"No data provided for {self.condition2}. Skipping...")

        # Load dataset 3 if provided
        if self.file_path3:
            self.df3 = pd.read_csv(self.file_path3, header=1).iloc[1:].reset_index(drop=True)
            self.df3 = self.df3.rename(columns={'Unnamed: 0': 'TF', 'Unnamed: 1': 'MotifID'})
            print(f"Loaded {self.condition3} with {len(self.df3)} rows.")
        else:
            print(f"No data provided for {self.condition3}. Skipping...")

        # Load gene expression data and apply filters
        print("Loading and filtering gene expression data...")
        self.expression_matrix = pd.read_csv(
            f"/work/project/ladcol_020/integration_visualization/{self.dataset_id}/{self.aliquot}/raw_count_matrix.txt",
            sep='\t', index_col=0
        )

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

        # Process dataset 1
        self.df1['TargetGeneCount'] = self.df1['TargetGenes'].apply(self._count_target_genes)
        self.df1['Condition'] = self.condition1

        # Process dataset 2 if available
        if self.df2 is not None:
            self.df2['TargetGeneCount'] = self.df2['TargetGenes'].apply(self._count_target_genes)
            self.df2['Condition'] = self.condition2

        # Process dataset 3 if available
        if self.df3 is not None:
            self.df3['TargetGeneCount'] = self.df3['TargetGenes'].apply(self._count_target_genes)
            self.df3['Condition'] = self.condition3

        print("Data preparation complete.")

    @staticmethod
    def _count_target_genes(target_genes_str):
        """
        Count the number of target genes in the target genes string.

        Parameters:
        -----------
        target_genes_str : str
            String representation of list of target genes

        Returns:
        --------
        int
            Number of target genes
        """
        if pd.isna(target_genes_str):
            return 0
        try:
            return len(eval(target_genes_str))  # Convert string to list
        except:
            return 0

    @staticmethod
    def _extract_gene_names(target_genes_str):
        """
        Extract gene names from the target genes string.

        Parameters:
        -----------
        target_genes_str : str
            String representation of list of target genes

        Returns:
        --------
        list
            List of gene names
        """
        if pd.isna(target_genes_str):
            return []
        try:
            target_genes_list = eval(target_genes_str)  # Convert string to list of tuples
            return [gene[0] for gene in target_genes_list]  # Extract gene names
        except:
            return []

    @staticmethod
    def _classify_cnv(cnv_value):
        """
        Classify CNV values into Loss, Neutral, or Gain categories.

        Parameters:
        -----------
        cnv_value : float
            CNV value

        Returns:
        --------
        str
            Classification as 'Loss', 'Neutral', or 'Gain'
        """
        if cnv_value < 0:
            return 'Loss'
        elif cnv_value == 0:
            return 'Neutral'
        else:
            return 'Gain'

    def _get_cnv_status(self, genes):
        """
        Get CNV status for a list of genes.

        Parameters:
        -----------
        genes : list
            List of gene names

        Returns:
        --------
        pandas.Series
            Series of CNV classifications for each gene
        """
        gene_cnv = self.cnv_matrix.loc[self.cnv_matrix.index.intersection(genes)]
        avg_cnv = gene_cnv.mean(axis=1)
        return avg_cnv.apply(self._classify_cnv)

    def plot_venn_diagram(self):
        print("Plotting triple Venn diagram...")
        tfs1 = set(self.df1['TF'])
        tfs2 = set(self.df2['TF'])
        tfs3 = set(self.df3['TF'])

        plt.figure(figsize=(10, 8))
        venn = venn3(
            [tfs1, tfs2, tfs3],
            set_labels=(f'{self.condition1} Regulons', f'{self.condition2} Regulons', f'{self.condition3} Regulons'),
            set_colors=("#440154", "#2a788e", "#fde725")
        )
        plt.title(f'Overlap of {self.condition1}, {self.condition2}, and {self.condition3} SCENIC Derived Regulons',
                  fontsize=12)
        plt.savefig(os.path.join(self.output_dir, '_triple_tf_overlap_venn.png'), dpi=300)
        plt.close()

        print("Triple Venn diagram saved as 'triple_tf_overlap_venn.png'.")

    def prepare_expression_data(self):
        print("Preparing expression data...")
        target_genes1 = self.df1['TargetGenes'].apply(self._extract_gene_names).explode().dropna().unique()
        target_genes2 = self.df2['TargetGenes'].apply(self._extract_gene_names).explode().dropna().unique()
        target_genes3 = self.df3['TargetGenes'].apply(self._extract_gene_names).explode().dropna().unique()

        expression1 = self.expression_matrix.loc[self.expression_matrix.index.intersection(target_genes1)]
        expression2 = self.expression_matrix.loc[self.expression_matrix.index.intersection(target_genes2)]
        expression3 = self.expression_matrix.loc[self.expression_matrix.index.intersection(target_genes3)]

        expression1_zscore = expression1.apply(zscore, axis=1)
        expression2_zscore = expression2.apply(zscore, axis=1)
        expression3_zscore = expression3.apply(zscore, axis=1)

        expression1_flat = expression1_zscore.values.flatten()
        expression2_flat = expression2_zscore.values.flatten()
        expression3_flat = expression3_zscore.values.flatten()

        self.expression_data_combined = pd.DataFrame({
            'Condition': [self.condition1] * len(expression1_flat) + [self.condition2] * len(expression2_flat) + [
                self.condition3] * len(expression3_flat),
            'Expression': list(expression1_flat) + list(expression2_flat) + list(expression3_flat)
        })

        print(
            f"Prepared expression data for {len(expression1_flat)} genes in {self.condition1}, {len(expression2_flat)} genes in {self.condition2}, and {len(expression3_flat)} genes in {self.condition3}.")

    def prepare_cnv_data(self):
        print("Preparing CNV data...")
        target_genes1 = self.df1['TargetGenes'].apply(self._extract_gene_names).explode().dropna().unique()
        target_genes2 = self.df2['TargetGenes'].apply(self._extract_gene_names).explode().dropna().unique()
        target_genes3 = self.df3['TargetGenes'].apply(self._extract_gene_names).explode().dropna().unique()

        cnv1 = self.cnv_matrix.loc[self.cnv_matrix.index.intersection(target_genes1)]
        cnv2 = self.cnv_matrix.loc[self.cnv_matrix.index.intersection(target_genes2)]
        cnv3 = self.cnv_matrix.loc[self.cnv_matrix.index.intersection(target_genes3)]

        cnv1_flat = cnv1.values.flatten()
        cnv2_flat = cnv2.values.flatten()
        cnv3_flat = cnv3.values.flatten()

        self.cnv_data_combined = pd.DataFrame({
            'Condition': [self.condition1] * len(cnv1_flat) + [self.condition2] * len(cnv2_flat) + [
                self.condition3] * len(cnv3_flat),
            'CNV': list(cnv1_flat) + list(cnv2_flat) + list(cnv3_flat)
        })

        print(
            f"Prepared CNV data for {len(cnv1_flat)} genes in {self.condition1}, {len(cnv2_flat)} genes in {self.condition2}, and {len(cnv3_flat)} genes in {self.condition3}.")

    def plot_violin_and_stacked_bar(self):
        print("Plotting violin and stacked bar plots...")
        violin_data = pd.DataFrame({
            'Condition': [self.condition1] * len(self.df1) + [self.condition2] * len(self.df2) + [
                self.condition3] * len(self.df3),
            'Target Gene Count': list(self.df1['TargetGeneCount']) + list(self.df2['TargetGeneCount']) + list(
                self.df3['TargetGeneCount'])
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

        # Create contingency tables
        self.contingency_tables[self.condition1] = self._create_tf_target_contingency_table(self.df1)
        chi2_1, p_value_1, dof_1, expected_1 = chi2_contingency(self.contingency_tables[self.condition1])
        self.chi2_results[self.condition1] = {
            'chi2': chi2_1,
            'p_value': p_value_1,
            'dof': dof_1,
            'expected': expected_1
        }

        if self.df2 is not None:
            self.contingency_tables[self.condition2] = self._create_tf_target_contingency_table(self.df2)
            chi2_2, p_value_2, dof_2, expected_2 = chi2_contingency(self.contingency_tables[self.condition2])
            self.chi2_results[self.condition2] = {
                'chi2': chi2_2,
                'p_value': p_value_2,
                'dof': dof_2,
                'expected': expected_2
            }

        if self.df3 is not None:
            self.contingency_tables[self.condition3] = self._create_tf_target_contingency_table(self.df3)
            chi2_3, p_value_3, dof_3, expected_3 = chi2_contingency(self.contingency_tables[self.condition3])
            self.chi2_results[self.condition3] = {
                'chi2': chi2_3,
                'p_value': p_value_3,
                'dof': dof_3,
                'expected': expected_3
            }

        # Create visualizations of the results
        # self.plot_contingency_cnv()
        self.plot_mosaic_cnv()

    def _create_tf_target_contingency_table(self, df):
        """
        Create a contingency table of TF CNV status vs target gene CNV status.

        Parameters:
        -----------
        df : pandas.DataFrame
            Regulon dataframe

        Returns:
        --------
        pandas.DataFrame
            Contingency table of TF CNV status vs target gene CNV status,
            formatted correctly with TF as rows and TG as columns.
        """
        print("Creating TF-target contingency table...")

        # Define the contingency table structure
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

            # Get CNV status for TF and target genes
            tf_cnv_status = self._get_cnv_status([tf]).get(tf, 'Neutral')
            target_cnv_status = self._get_cnv_status(target_genes)

            # Count relationships
            for _, status in target_cnv_status.items():
                contingency_table.loc[tf_cnv_status, f'TG {status}'] += 1

        print(f"Total number of entries with no target genes: {empty_count}")

        return contingency_table

    def plot_contingency_cnv(self):
        """
        Create stacked bar plots of contingency tables colored by target gene CNV status.
        Significant residuals are marked with asterisks.
        """
        print("Creating stacked bar plots with CNV-based coloring...")

        # Define colors for CNV categories
        cnv_colors = sns.color_palette("Set2", n_colors=len(
            self.contingency_tables[list(self.contingency_tables.keys())[0]].columns))
        cnv_color_map = {col: cnv_colors[i] for i, col in
                         enumerate(self.contingency_tables[list(self.contingency_tables.keys())[0]].columns)}

        # Create a figure for each condition
        for condition, contingency_table in self.contingency_tables.items():
            chi2_result = self.chi2_results[condition]
            expected = chi2_result['expected']

            # Calculate Pearson residuals
            observed = contingency_table.values
            residuals = (observed - expected) / np.sqrt(expected)

            # Convert to proportions for plotting (each row sums to 1)
            proportions = contingency_table.div(contingency_table.sum(axis=1), axis=0)

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Setup for stacked bar
            bottom = np.zeros(len(proportions))
            width = 0.6
            x = np.arange(len(proportions.index))

            # Loop through each column (target CNV status)
            for i, col in enumerate(proportions.columns):
                values = proportions[col].values
                col_residuals = residuals[:, i]
                color = cnv_color_map[col]  # Assign color based on CNV status

                for j, (value, residual) in enumerate(zip(values, col_residuals)):
                    rect = ax.bar(x[j], value, width, bottom=bottom[j], color=color,
                                  edgecolor='white', linewidth=1)

                    # Center of the bar for text positioning
                    x_pos = x[j]
                    y_pos = bottom[j] + value / 2

                    # Add percentage text
                    percentage = f"{value:.2f}"
                    ax.text(x_pos, y_pos, percentage, ha='center', va='center',
                            color='black', fontsize=9)

                    # Add asterisks for significant residuals
                    if abs(residual) > 1.96:
                        stars = '*' if abs(residual) > 1.96 else ''
                        stars += '*' if abs(residual) > 2.58 else ''  # 99% confidence
                        stars += '*' if abs(residual) > 3.29 else ''  # 99.9% confidence
                        ax.text(x_pos, bottom[j] + value + 0.01, stars, ha='center', va='bottom')

                bottom += values

            # Add labels and legend
            ax.set_title(f'TF-Target Gene CNV Relationships - {condition}')
            ax.set_xticks(x)
            ax.set_xticklabels(proportions.index)
            ax.set_xlabel('TF CNV Status')
            ax.set_ylabel('Proportion of Target Genes')

            # Add legend for CNV status
            legend_elements = [plt.Rectangle((0, 0), 1, 1, color=cnv_color_map[col], label=col)
                               for col in proportions.columns]
            ax.legend(handles=legend_elements, title='Target CNV Status', loc='upper right')

            # Add chi-square test results as text
            plt.figtext(0.01, 0.01,
                        f"Chi-square: {chi2_result['chi2']:.2f}, p-value: {chi2_result['p_value']:.4f}, df: {chi2_result['dof']}",
                        ha='left')

            # Adjust layout and save
            plt.tight_layout()
            filename = f"{condition.lower()}_cnv_plot.png"
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            plt.close()

            print(f"Saved stacked bar plot for {condition} as {filename}")

    def plot_mosaic_cnv(self):
        """
        Create mosaic plots of contingency tables colored by Pearson residuals.
        Asterisks denote significance levels inside each block.
        """
        print("Creating mosaic plots with Pearson residual-based coloring...")

        cmap = plt.cm.RdBu_r  # Colormap for residuals

        for condition, contingency_table in self.contingency_tables.items():
            chi2_result = self.chi2_results[condition]
            expected = chi2_result['expected']

            observed = contingency_table.values
            residuals = (observed - expected) / np.sqrt(expected)

            # Dynamically adjust color scale based on residuals
            abs_max_residual = np.max(np.abs(residuals))
            norm = plt.Normalize(-abs_max_residual, abs_max_residual)

            # Define row and column labels
            row_labels = ['TF Loss', 'TF Neutral', 'TF Gain']
            col_labels = ['TG Loss', 'TG Neutral', 'TG Gain']

            # Ensure proper labeling
            contingency_table.index = row_labels
            contingency_table.columns = col_labels

            # Convert to proportions for visualization
            proportions = contingency_table.div(contingency_table.sum(axis=1), axis=0)

            # Prepare data for mosaic plot
            mosaic_data = {}
            labelizer_dict = {}

            for i, row_label in enumerate(row_labels):
                for j, col_label in enumerate(col_labels):
                    mosaic_data[(row_label, col_label)] = proportions.iloc[i, j]

                    # Determine significance level for the block
                    residual = residuals[i, j]
                    if abs(residual) > 2.58:
                        labelizer_dict[(row_label, col_label)] = "***"
                    elif abs(residual) > 1.96:
                        labelizer_dict[(row_label, col_label)] = "**"
                    elif abs(residual) > 1.645:
                        labelizer_dict[(row_label, col_label)] = "*"
                    else:
                        labelizer_dict[(row_label, col_label)] = ""  # No label if not significant

            # Define properties function to color tiles
            def props(key):
                residual = residuals[row_labels.index(key[0]), col_labels.index(key[1])]
                return {"color": cmap(norm(residual))}

            # Convert labelizer_dict into a callable function with larger font size
            labelizer_func = lambda key: f"{labelizer_dict.get(key, '')}"  # Move asterisks down slightly

            # Create the mosaic plot
            fig, ax = plt.subplots(figsize=(8, 6))
            mosaic(mosaic_data, title=f'TF-Target Gene CNV Relationships - {condition}',
                   properties=props, labelizer=labelizer_func, ax=ax)

            # Add colorbar further to the right and slightly smaller
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.08)  # Move colorbar slightly right
            cbar.set_label('Pearson Residual')

            # Format p-value in scientific notation closer to the plot
            p_value = chi2_result['p_value']
            plt.figtext(0.9, 0.08, f"p = {p_value:.3e}", ha='center', fontsize=12)  # Moved p-value up

            # Save without using tight layout
            filename = f"{condition.lower()}_cnv_mosaic.png"
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            plt.close()

            print(f"Saved mosaic plot for {condition} as {filename}")

    def analyze(self):
        """
        Run the complete analysis pipeline and print summary statistics.
        """
        print("Starting analysis...\n")
        self.load_data()
        self.prepare_data()
        self.prepare_expression_data()
        self.prepare_cnv_data()
        self.plot_venn_diagram()
        # self.plot_violin_and_stacked_bar()
        self.perform_statistical_tests()

        print("\nContingency Tables\n" + "-" * 40)
        print("\nDataset:", self.condition1)
        print(self.contingency_tables[self.condition1], "\n")

        print("\nDataset:", self.condition2)
        print(self.contingency_tables[self.condition2], "\n")

        print("\nDataset:", self.condition3)
        print(self.contingency_tables[self.condition3], "\n")

        print("\nChi-Square Test Results\n" + "-" * 40)
        for label, result in self.chi2_results.items():
            print(f"Dataset: {label}")
            print(f"   - Chi2 Value: {result['chi2']:.3f}")
            print(f"   - p-value: {result['p_value']:.5f}")
            print(f"   - Degrees of Freedom: {result['dof']}")
            print("   - Expected Frequencies:\n", pd.DataFrame(result['expected']), "\n")


if __name__ == "__main__":
    # Hardcoded paths for now
    file_path1 = '/work/project/ladcol_020/scGRNi/RNA/SCENIC/ccRCC_GBM/C3N-00495-T1_CPT0078510004/reg.csv'
    file_path2 = '/work/project/ladcol_020/scGRNi/RNA/SCENIC/ccRCC_GBM/C3N-00495-T1_CPT0078510004/Tumor_reg.csv'
    file_path3 = '/work/project/ladcol_020/scGRNi/RNA/SCENIC/ccRCC_GBM/C3N-00495-T1_CPT0078510004/Non-Tumor_reg.csv'

    # Optional names
    condition1 = "whole_dataset"
    condition2 = "Tumor"
    condition3 = "Non-Tumor"

    dataset_id = "ccRCC_GBM"
    aliquot = "C3N-00495-T1_CPT0078510004" #C3N-00495-T1_CPT0078510004 #C3L-00004-T1_CPT0001540013

    comparison = RegfileComparison(file_path1, condition1, dataset_id, aliquot, file_path2, file_path3,
                                   condition2, condition3)

    comparison.analyze()