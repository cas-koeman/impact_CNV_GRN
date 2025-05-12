import os
import loompy as lp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class RegulonAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.subcluster_dirs = self._find_subcluster_dirs()
        self.regulon_data = self._load_all_regulon_data()

    def create_heatmap(self, data_df, title, output_file=None, output_png=None, cmap="viridis", 
                    vmin=None, vmax=None, annot=True, fmt=".2f", figsize=(10, 8), triangle="lower"):
        """
        Create a triangular heatmap visualization for correlation or similarity matrices.

        Parameters:
        -----------
        data_df : pandas DataFrame
            DataFrame containing the data to visualize.
        title : str
            Title for the heatmap.
        output_file : str, optional
            Path to save the output as a PDF file.
        output_png : str, optional
            Path to save the output as a PNG file.
        cmap : str, optional
            Colormap for the heatmap.
        vmin, vmax : float, optional
            Minimum and maximum values for color scaling.
        annot : bool, optional
            Whether to annotate the heatmap with data values.
        fmt : str, optional
            Format for the annotations.
        figsize : tuple, optional
            Figure size in inches.
        triangle : str, optional
            Which triangle to display: 'lower' or 'upper'.

        Returns:
        --------
        fig, ax : matplotlib figure and axis objects.
        """
        plt.figure(figsize=figsize)

        # Generate a mask for the upper or lower triangle (excluding the diagonal)
        mask = np.zeros_like(data_df, dtype=bool)
        if triangle == "upper":
            mask[np.tril_indices_from(mask, k=-1)] = True  # keep diagonal
        elif triangle == "lower":
            mask[np.triu_indices_from(mask, k=1)] = True  # keep diagonal

        cbar_label = "Pearson coefficient" if "orrelation" in title.lower() else "Jaccard index"

        ax = sns.heatmap(
            data_df,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            mask=mask,
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': cbar_label}
        )

        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
        if output_png:
            plt.savefig(output_png, bbox_inches='tight', dpi=300)

        fig = plt.gcf()
        plt.close()

        return fig, ax
        
    def _find_subcluster_dirs(self):
        """Find all Tumor subcluster directories"""
        subclusters = []
        for item in os.listdir(self.base_dir):
            if item.startswith('Tumor_s') and os.path.isdir(os.path.join(self.base_dir, item)):
                subclusters.append(item)
        return sorted(subclusters)
    
    def _load_all_regulon_data(self):
        """Load regulon data from all subclusters"""
        regulon_data = {}
        for subcluster in self.subcluster_dirs:
            loom_path = os.path.join(self.base_dir, subcluster, f"{subcluster}_pyscenic_output.loom")
            if os.path.exists(loom_path):
                regulon_data[subcluster] = self._extract_regulons(loom_path)
        return regulon_data
    
    def _extract_regulons(self, loom_path):
        """Extract regulons and their target genes from a loom file"""
        with lp.connect(loom_path, mode='r', validate=False) as lf:
            regulons = {
                'names': lf.ca['RegulonsAUC'].dtype.names,
                'target_genes': self._get_target_genes(lf)
            }
        return regulons
    
    def _get_target_genes(self, loom_connection):
        """Extract target genes for each regulon"""
        regulons_binary = np.array(loom_connection.ra['Regulons'].tolist())
        target_genes = {}
        for i, name in enumerate(loom_connection.ca['RegulonsAUC'].dtype.names):
            target_genes[name] = list(loom_connection.ra['Gene'][np.where(regulons_binary[:, i] == 1)[0]])
        return target_genes
    
    def analyze_regulon_overlap(self, output_dir):
        """Analyze regulon overlap between subclusters with heatmaps"""
        all_regulons = set()
        for data in self.regulon_data.values():
            all_regulons.update(data['names'])
        all_regulons = sorted(all_regulons)
        
        presence_matrix = []
        for subcluster, data in self.regulon_data.items():
            presence_matrix.append([1 if reg in data['names'] else 0 for reg in all_regulons])
        
        subcluster_names = list(self.regulon_data.keys())
        overlap_matrix = np.zeros((len(subcluster_names), len(subcluster_names)), dtype=int)
        
        for i, sc1 in enumerate(subcluster_names):
            for j, sc2 in enumerate(subcluster_names):
                if i <= j:
                    regulons1 = set(self.regulon_data[sc1]['names'])
                    regulons2 = set(self.regulon_data[sc2]['names'])
                    overlap = len(regulons1 & regulons2)
                    overlap_matrix[i, j] = overlap
                    overlap_matrix[j, i] = overlap
        
        overlap_df = pd.DataFrame(overlap_matrix, 
                                index=subcluster_names, 
                                columns=subcluster_names)
        
        proportional_matrix = np.zeros_like(overlap_matrix, dtype=float)
        for i, sc1 in enumerate(subcluster_names):
            for j, sc2 in enumerate(subcluster_names):
                if i == j:
                    proportional_matrix[i, j] = 1.0
                else:
                    regulons1 = set(self.regulon_data[sc1]['names'])
                    regulons2 = set(self.regulon_data[sc2]['names'])
                    union = len(regulons1 | regulons2)
                    if union > 0:
                        proportional_matrix[i, j] = overlap_matrix[i, j] / union
        
        proportion_df = pd.DataFrame(proportional_matrix, 
                                    index=subcluster_names, 
                                    columns=subcluster_names)
        
        # Calculate Pearson correlation matrix for regulons
        correlation_matrix = np.zeros_like(proportional_matrix)
        for i, sc1 in enumerate(subcluster_names):
            for j, sc2 in enumerate(subcluster_names):
                # Create binary vectors for regulons presence
                all_regs = sorted(list(set(self.regulon_data[sc1]['names']) | set(self.regulon_data[sc2]['names'])))
                vec1 = np.array([1 if reg in self.regulon_data[sc1]['names'] else 0 for reg in all_regs])
                vec2 = np.array([1 if reg in self.regulon_data[sc2]['names'] else 0 for reg in all_regs])
                
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Calculate Pearson correlation
                    correlation_matrix[i, j] = np.corrcoef(vec1, vec2)[0, 1]
        
        correlation_df = pd.DataFrame(correlation_matrix,
                                    index=subcluster_names,
                                    columns=subcluster_names)
        
        print("Regulon overlap matrix values (absolute counts):")
        print(overlap_df)
        print("\nRegulon overlap matrix values (proportions - Jaccard index):")
        print(proportion_df)
        print("\nRegulon correlation matrix (Pearson):")
        print(correlation_df)
        
        # Create heatmaps for overlap and correlation matrices
        # Heatmap for absolute overlap
        overlap_heatmap_file = os.path.join(output_dir, "regulon_overlap_heatmap.pdf")
        overlap_heatmap_png = os.path.join(output_dir, "regulon_overlap_heatmap.png")
        self.create_heatmap(
            overlap_df, 
            "Regulon Overlap Between Subclusters (Absolute Counts)",
            output_file=overlap_heatmap_file,
            output_png=overlap_heatmap_png,
            cmap="YlOrRd",
            annot=True,
            fmt="d"
        )
        
        # Heatmap for Jaccard similarity
        jaccard_heatmap_file = os.path.join(output_dir, "regulon_jaccard_heatmap.pdf")
        jaccard_heatmap_png = os.path.join(output_dir, "regulon_jaccard_heatmap.png")
        self.create_heatmap(
            proportion_df, 
            "Regulon Similarity Between Subclusters (Jaccard Index)",
            output_file=jaccard_heatmap_file,
            output_png=jaccard_heatmap_png,
            cmap="viridis",
            vmin=0,
            vmax=1
        )
        
        # Heatmap for Pearson correlation
        corr_heatmap_file = os.path.join(output_dir, "regulon_correlation_heatmap.pdf")
        corr_heatmap_png = os.path.join(output_dir, "regulon_correlation_heatmap.png")
        self.create_heatmap(
            correlation_df, 
            "Regulon Correlation Between Subclusters (Pearson)",
            output_file=corr_heatmap_file,
            output_png=corr_heatmap_png,
            cmap="coolwarm",
            vmin=-1,
            vmax=1
        )
        
        # Save dataframes
        overlap_df.to_csv(os.path.join(output_dir, "regulon_overlap_matrix.csv"))
        proportion_df.to_csv(os.path.join(output_dir, "regulon_jaccard_matrix.csv"))
        correlation_df.to_csv(os.path.join(output_dir, "regulon_correlation_matrix.csv"))
        
        return overlap_df, proportion_df, correlation_df
    
    def analyze_target_genes(self, output_dir):
        """Analyze target gene overlap between subclusters with heatmaps"""
        subcluster_names = list(self.regulon_data.keys())
        all_subcluster_targets = {}
        
        for sc in subcluster_names:
            all_targets = set()
            for regulon, targets in self.regulon_data[sc]['target_genes'].items():
                all_targets.update(targets)
            all_subcluster_targets[sc] = all_targets
            print(f"Subcluster {sc}: {len(all_targets)} total target genes")
        
        overlap_matrix = np.zeros((len(subcluster_names), len(subcluster_names)), dtype=int)
        
        for i, sc1 in enumerate(subcluster_names):
            for j, sc2 in enumerate(subcluster_names):
                if i <= j:
                    targets1 = all_subcluster_targets[sc1]
                    targets2 = all_subcluster_targets[sc2]
                    overlap = len(targets1 & targets2)
                    overlap_matrix[i, j] = overlap
                    overlap_matrix[j, i] = overlap
        
        overlap_df = pd.DataFrame(overlap_matrix, 
                                 index=subcluster_names, 
                                 columns=subcluster_names)
        
        proportional_matrix = np.zeros_like(overlap_matrix, dtype=float)
        for i, sc1 in enumerate(subcluster_names):
            for j, sc2 in enumerate(subcluster_names):
                if i == j:
                    proportional_matrix[i, j] = 1.0
                else:
                    targets1 = all_subcluster_targets[sc1]
                    targets2 = all_subcluster_targets[sc2]
                    union = len(targets1 | targets2)
                    if union > 0:
                        proportional_matrix[i, j] = overlap_matrix[i, j] / union
        
        proportion_df = pd.DataFrame(proportional_matrix, 
                                    index=subcluster_names, 
                                    columns=subcluster_names)
        
        # Calculate Pearson correlation matrix for target genes
        correlation_matrix = np.zeros_like(proportional_matrix)
        for i, sc1 in enumerate(subcluster_names):
            for j, sc2 in enumerate(subcluster_names):
                # Create binary vectors for all possible target genes
                all_genes = sorted(list(set(all_subcluster_targets[sc1]) | set(all_subcluster_targets[sc2])))
                vec1 = np.array([1 if gene in all_subcluster_targets[sc1] else 0 for gene in all_genes])
                vec2 = np.array([1 if gene in all_subcluster_targets[sc2] else 0 for gene in all_genes])
                
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Calculate Pearson correlation
                    correlation_matrix[i, j] = np.corrcoef(vec1, vec2)[0, 1]
        
        correlation_df = pd.DataFrame(correlation_matrix,
                                     index=subcluster_names,
                                     columns=subcluster_names)
        
        print("\nTotal target gene overlap matrix (absolute counts):")
        print(overlap_df)
        print("\nTotal target gene overlap matrix (proportions - Jaccard index):")
        print(proportion_df)
        print("\nTarget gene correlation matrix (Pearson):")
        print(correlation_df)
        
        # Create heatmaps for target gene data
        # Heatmap for absolute overlap
        overlap_heatmap_file = os.path.join(output_dir, "target_gene_overlap_heatmap.pdf")
        overlap_heatmap_png = os.path.join(output_dir, "target_gene_overlap_heatmap.png")
        self.create_heatmap(
            overlap_df, 
            "Target Gene Overlap Between Subclusters (Absolute Counts)",
            output_file=overlap_heatmap_file,
            output_png=overlap_heatmap_png,
            cmap="YlOrRd",
            annot=True,
            fmt="d"
        )
        
        # Heatmap for Jaccard similarity
        jaccard_heatmap_file = os.path.join(output_dir, "target_gene_jaccard_heatmap.pdf")
        jaccard_heatmap_png = os.path.join(output_dir, "target_gene_jaccard_heatmap.png")
        self.create_heatmap(
            proportion_df, 
            "Target Gene Similarity Between Subclusters (Jaccard Index)",
            output_file=jaccard_heatmap_file,
            output_png=jaccard_heatmap_png,
            cmap="viridis",
            vmin=0,
            vmax=1
        )
        
        # Heatmap for Pearson correlation
        corr_heatmap_file = os.path.join(output_dir, "target_gene_correlation_heatmap.pdf")
        corr_heatmap_png = os.path.join(output_dir, "target_gene_correlation_heatmap.png")
        self.create_heatmap(
            correlation_df, 
            "Target Gene Correlation Between Subclusters (Pearson)",
            output_file=corr_heatmap_file,
            output_png=corr_heatmap_png,
            cmap="coolwarm",
            vmin=-1,
            vmax=1
        )
        
        # Save dataframes
        overlap_df.to_csv(os.path.join(output_dir, "target_gene_overlap_matrix.csv"))
        proportion_df.to_csv(os.path.join(output_dir, "target_gene_jaccard_matrix.csv"))
        correlation_df.to_csv(os.path.join(output_dir, "target_gene_correlation_matrix.csv"))
        
        summary_data = []
        for i, sc1 in enumerate(subcluster_names):
            row = {'Subcluster': sc1, 'Total_Target_Genes': len(all_subcluster_targets[sc1])}
            for j, sc2 in enumerate(subcluster_names):
                if i != j:
                    row[f'Shared_with_{sc2}'] = overlap_matrix[i, j]
                    row[f'Jaccard_with_{sc2}'] = proportional_matrix[i, j]
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, "target_gene_summary.csv"))
        
        return overlap_df, proportion_df, correlation_df, summary_df, all_subcluster_targets

def analyze_intratumor_comparisons(sample_ids, dataset_id, output_base_dir):
    """Analyze intratumor comparisons for all samples"""
    base_scenic_dir = "/work/project/ladcol_020/scGRNi/RNA/SCENIC"
    
    for sample_id in sample_ids:
        sample_path = os.path.join(base_scenic_dir, dataset_id, sample_id)
        if not os.path.exists(sample_path):
            print(f"Warning: Sample directory not found - {sample_path}")
            continue
            
        print(f"\n{'='*80}")
        print(f"ANALYZING SAMPLE: {sample_id}")
        print(f"{'='*80}")
        
        # Create output directory for this sample
        sample_output_dir = os.path.join(output_base_dir, sample_id)
        os.makedirs(sample_output_dir, exist_ok=True)
        
        # Initialize analyzer for this sample
        analyzer = RegulonAnalyzer(sample_path)
        
        # Perform regulon overlap analysis
        print("\nAnalyzing regulon overlap between subclusters...")
        analyzer.analyze_regulon_overlap(sample_output_dir)
        
        # Perform target gene analysis
        print("\nAnalyzing target gene overlap between subclusters...")
        analyzer.analyze_target_genes(sample_output_dir)
        
        print(f"\nCompleted analysis for sample {sample_id}. Results saved to: {sample_output_dir}")

def main():
    """Run intratumor comparison analysis for all samples"""
    dataset_id = "ccRCC_GBM"
    output_base_dir = "/work/project/ladcol_020/integration_GRN_CNV/scenic_analysis/intratumor_comp"
    
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
    
    # Create output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Analyze all samples
    analyze_intratumor_comparisons(sample_ids, dataset_id, output_base_dir)
    
    print("\n[Complete] All intratumor analyses finished with outputs saved to:", output_base_dir)

if __name__ == "__main__":
    main()