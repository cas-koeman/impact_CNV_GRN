import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from typing import Dict, List
import loompy as lp

class GeneExpressionCNVAnalyzer:
    def __init__(self, target_gene, datasets, cnv_dir, raw_count_dir, regulon_dir=None):
        self.target_gene = target_gene
        self.datasets = datasets
        self.cnv_dir = cnv_dir
        self.raw_count_dir = raw_count_dir
        self.regulon_dir = regulon_dir
        self.results = []
        self.selected_datasets = []
        self.common_regulons = None

    def analyze_gene(self):
        print(f"\nAnalyzing gene {self.target_gene} across {len(self.datasets)} tumor samples...")

        for dataset in self.datasets:
            try:
                cnv_matrix_path = os.path.join(self.cnv_dir, dataset, "cnv_matrix.tsv")
                raw_count_path = os.path.join(self.raw_count_dir, dataset, "raw_count_matrix.txt")

                if not os.path.exists(cnv_matrix_path):
                    print(f"Warning: CNV matrix file not found for {dataset}, skipping.")
                    continue

                if not os.path.exists(raw_count_path):
                    print(f"Warning: Raw count matrix file not found for {dataset}, skipping.")
                    continue

                print(f"Loading data for {dataset}...")
                cnv_matrix = pd.read_csv(cnv_matrix_path, sep='\t', index_col=0)
                raw_count_matrix = pd.read_csv(raw_count_path, sep='\t', index_col=0)

                # Print metrics before filtering
                print("\nExpression Matrix Metrics Before Filtering:")
                print(f"Total genes: {raw_count_matrix.shape[0]}")
                print(f"Total cells: {raw_count_matrix.shape[1]}")
                print(f"Average counts per gene: {raw_count_matrix.mean(axis=1).mean():.2f}")
                print(f"Average counts per cell: {raw_count_matrix.mean(axis=0).mean():.2f}")
                print(f"Percentage of zeros: {(raw_count_matrix == 0).mean().mean()*100:.2f}%")
                print(f"Genes with any expression: {(raw_count_matrix > 0).any(axis=1).sum()}")
                print(f"Cells with any expression: {(raw_count_matrix > 0).any(axis=0).sum()}\n")

                cnv_matrix = cnv_matrix * 2 - 2  # Convert to copy number change

                print(f"CNV matrix dimensions: {cnv_matrix.shape}")
                print(f"Raw count matrix dimensions (before filtering): {raw_count_matrix.shape}")

                gene_filter = (raw_count_matrix > 0).sum(axis=1) >= 200
                filtered_matrix = raw_count_matrix[gene_filter]
                cell_filter = (filtered_matrix > 0).sum(axis=0) >= 3
                filtered_matrix = filtered_matrix.loc[:, cell_filter]

                print(f"Filtered raw count matrix dimensions: {filtered_matrix.shape}")

                z_score_matrix = filtered_matrix.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
                print(f"Z-score matrix dimensions: {z_score_matrix.shape}")

                if self.target_gene not in cnv_matrix.index:
                    print(f"Warning: Gene {self.target_gene} not found in CNV matrix for {dataset}, skipping.")
                    continue

                if self.target_gene not in z_score_matrix.index:
                    print(f"Warning: Gene {self.target_gene} not found in filtered expression matrix for {dataset}, skipping.")
                    continue

                gene_cnv = cnv_matrix.loc[self.target_gene]
                gene_z_scores = z_score_matrix.loc[self.target_gene]

                common_cells = gene_cnv.index.intersection(gene_z_scores.index)

                if len(common_cells) == 0:
                    print(f"Warning: No common cells found between CNV and expression data for {dataset}, skipping.")
                    continue

                gene_cnv = gene_cnv[common_cells]
                gene_z_scores = gene_z_scores[common_cells]

                avg_z_score = np.mean(gene_z_scores)
                avg_cnv = np.mean(gene_cnv)

                self.results.append({
                    'Tumor Sample': dataset,
                    'Avg Z-Score Expression': avg_z_score,
                    'Avg Copy Number': avg_cnv,
                    'Cell Count': len(common_cells),
                    'CNV Matrix': cnv_matrix,
                    'Z-Score Matrix': z_score_matrix,
                    'Common Cells': common_cells
                })

                print(f"Processed {dataset}: Avg Z-Score = {avg_z_score:.4f}, Avg CNV = {avg_cnv:.4f}, Cells = {len(common_cells)}")

            except Exception as e:
                print(f"Error processing dataset {dataset}: {str(e)}")

    def select_non_zero_datasets(self):
        if not self.results:
            print(f"No results found for gene {self.target_gene}.")
            return None

        results_df = pd.DataFrame([{k: v for k, v in result.items() if k not in ['CNV Matrix', 'Z-Score Matrix', 'Common Cells']}
                                for result in self.results])

        selected_df = results_df[(results_df['Avg Z-Score Expression'] != 0) & (results_df['Avg Copy Number'] != 0)]

        if selected_df.empty:
            print("No datasets found where both average z-score and average CNV are non-zero.")
            return None

        self.selected_datasets = [
            result for result in self.results
            if result['Tumor Sample'] in selected_df['Tumor Sample'].values
        ]

        print(f"\nSelected {len(self.selected_datasets)} datasets with non-zero values:")
        for dataset in self.selected_datasets:
            print(f"  - {dataset['Tumor Sample']}: Avg Z-Score = {dataset['Avg Z-Score Expression']:.4f}, Avg CNV = {dataset['Avg Copy Number']:.4f}")

        return selected_df

    def load_regulons(self):
        if not self.selected_datasets or not self.regulon_dir:
            print("No selected datasets or regulon directory not specified.")
            return None

        dataset_regulons = {}

        for dataset_info in self.selected_datasets:
            dataset = dataset_info['Tumor Sample']
            loom_path = os.path.join(self.regulon_dir, "ccRCC_GBM", dataset, "Tumor_pyscenic_output.loom")

            if not os.path.exists(loom_path):
                print(f"Warning: SCENIC loom file not found at {loom_path}, skipping.")
                continue

            try:
                with lp.connect(loom_path, mode='r', validate=False) as lf:
                    regulon_names = lf.ca['RegulonsAUC'].dtype.names
                    tf_names = [name.split('(+)')[0] if '(+)' in name else name for name in regulon_names]
                    regulons_binary = np.array(lf.ra['Regulons'].tolist())
                    target_genes = {}

                    for i, name in enumerate(regulon_names):
                        target_indices = np.where(regulons_binary[:, i] == 1)[0]
                        genes = list(lf.ra['Gene'][target_indices])
                        tf_name = tf_names[i]
                        if self.target_gene in genes:
                            target_genes[tf_name] = genes

                dataset_regulons[dataset] = target_genes
                print(f"Loaded {len(target_genes)} regulons for {dataset} that include {self.target_gene}")

            except Exception as e:
                print(f"Error loading regulons for dataset {dataset}: {str(e)}")

        if not dataset_regulons:
            print("No regulon data could be loaded.")
            return None

        common_regulons = set(dataset_regulons[list(dataset_regulons.keys())[0]].keys())
        for dataset, regulons in dataset_regulons.items():
            common_regulons &= set(regulons.keys())

        self.common_regulons = list(common_regulons)

        if not self.common_regulons:
            print(f"No common regulons found across all selected datasets for target gene {self.target_gene}.")
            return None

        print(f"\nFound {len(self.common_regulons)} common regulons across all selected datasets:")
        for tf in self.common_regulons:
            print(f"  - {tf}")

        return self.common_regulons

    def analyze_tf_tg_relationship(self, transcription_factor):
        if not self.selected_datasets:
            print("No datasets selected for analysis.")
            return None

        print(f"\nAnalyzing TF-TG relationship: {transcription_factor} -> {self.target_gene}")

        all_cell_data = []

        for dataset_info in self.selected_datasets:
            dataset = dataset_info['Tumor Sample']
            cnv_matrix = dataset_info['CNV Matrix']
            z_score_matrix = dataset_info['Z-Score Matrix']
            common_cells = dataset_info['Common Cells']

            if transcription_factor not in cnv_matrix.index:
                print(f"Warning: TF {transcription_factor} not found in CNV matrix for {dataset}, skipping.")
                continue

            if transcription_factor not in z_score_matrix.index:
                print(f"Warning: TF {transcription_factor} not found in filtered expression matrix for {dataset}, skipping.")
                continue

            tf_cnv = cnv_matrix.loc[transcription_factor][common_cells]
            tf_z_scores = z_score_matrix.loc[transcription_factor][common_cells]
            tg_cnv = cnv_matrix.loc[self.target_gene][common_cells]
            tg_z_scores = z_score_matrix.loc[self.target_gene][common_cells]

            cell_data = pd.DataFrame({
                'Dataset': dataset,
                'Cell': common_cells,
                'TF_Expression': tf_z_scores.values,
                'TG_Expression': tg_z_scores.values,
                'TF_CNV': tf_cnv.values,
                'TG_CNV': tg_cnv.values
            })

            all_cell_data.append(cell_data)

        if not all_cell_data:
            print(f"No valid data found for TF {transcription_factor} and TG {self.target_gene}.")
            return None

        combined_cell_data = pd.concat(all_cell_data, ignore_index=True)
        print(combined_cell_data.head(20))
        self.plot_tf_tg_relationship(combined_cell_data, transcription_factor)
        return combined_cell_data

    def plot_tf_tg_relationship(self, cell_data, transcription_factor):
        cell_data['TF_CNV_Status'] = np.select(
            [cell_data['TF_CNV'] > 0, cell_data['TF_CNV'] < 0],
            ['Gain', 'Loss'],
            default='Neutral'
        )
        cell_data['TG_CNV_Status'] = np.select(
            [cell_data['TG_CNV'] > 0, cell_data['TG_CNV'] < 0],
            ['Gain', 'Loss'],
            default='Neutral'
        )

        cell_data['CNV_Combination'] = (
            "TF_" + cell_data['TF_CNV_Status'] + 
            "/TG_" + cell_data['TG_CNV_Status']
        )

        palette = {
            'TF_Loss/TG_Loss': '#d62728',
            'TF_Loss/TG_Neutral': '#ff7f0e',
            'TF_Loss/TG_Gain': '#f7b6d2',
            'TF_Neutral/TG_Loss': '#1f77b4',
            'TF_Neutral/TG_Neutral': '#7f7f7f',
            'TF_Neutral/TG_Gain': '#2ca02c',
            'TF_Gain/TG_Loss': '#9467bd',
            'TF_Gain/TG_Neutral': '#8c564b',
            'TF_Gain/TG_Gain': '#e377c2'
        }

        plt.figure(figsize=(12, 8))
        
        sns.scatterplot(
            data=cell_data,
            x='TF_Expression',
            y='TG_Expression',
            hue='CNV_Combination',
            palette=palette,
            alpha=0.7,
            s=50,
            edgecolor='black',
            linewidth=0.3
        )

        plt.title(f'{transcription_factor} vs {self.target_gene} Expression\nColored by CNV Status', pad=20)
        plt.xlabel(f'{transcription_factor} Expression (Z-score)')
        plt.ylabel(f'{self.target_gene} Expression (Z-score)')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        plt.legend(
            title='CNV Combination',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )

        corr, pval = stats.pearsonr(cell_data['TF_Expression'], cell_data['TG_Expression'])
        plt.annotate(
            f"Pearson r = {corr:.2f} (p = {pval:.3f})\nTotal cells = {len(cell_data)}",
            xy=(0.02, 0.95),
            xycoords='axes fraction',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8)
        )

        plt.tight_layout()
        
        plot_file = f"{self.target_gene}_{transcription_factor}_CNV_plot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_file}")
        plt.show()


def main():
    target_gene = input("Enter the target gene to analyze (default: CXCR4): ") or "CXCR4"
    sample_ids = ["C3L-00004-T1_CPT0001540013"]
    
    base_dir = "/work/project/ladcol_020/integration_GRN_CNV/"
    cnv_dir = os.path.join(base_dir, "ccRCC_GBM")
    expr_dir = os.path.join(base_dir, "ccRCC_GBM")
    regulon_dir = "/work/project/ladcol_020/scGRNi/RNA/SCENIC/"

    analyzer = GeneExpressionCNVAnalyzer(
        target_gene=target_gene,
        datasets=sample_ids,
        cnv_dir=cnv_dir,
        raw_count_dir=expr_dir,
        regulon_dir=regulon_dir
    )

    analyzer.analyze_gene()
    selected_df = analyzer.select_non_zero_datasets()

    if selected_df is None or selected_df.empty:
        print("No suitable datasets found. Exiting.")
        return

    common_regulons = analyzer.load_regulons()

    if not common_regulons:
        print("No common regulons found. Continuing with manual TF selection.")
        tf = input("Enter a transcription factor to analyze: ")
        if tf:
            analyzer.analyze_tf_tg_relationship(tf)
    else:
        for tf in common_regulons:
            analyzer.analyze_tf_tg_relationship(tf)
            if len(common_regulons) > 1 and tf != common_regulons[-1]:
                continue_analysis = input(f"Continue with next TF? (y/n, default: y): ").lower() or 'y'
                if continue_analysis != 'y':
                    break

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()