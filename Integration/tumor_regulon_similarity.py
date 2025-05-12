import os
import loompy as lp
import pandas as pd
import numpy as np
from pycirclize import Circos
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class RegulonAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.subcluster_dirs = self._find_subcluster_dirs()
        self.regulon_data = self._load_all_regulon_data()
        
    def create_heatmap(self, data_df, title, output_file=None, output_png=None, cmap="viridis", 
                      vmin=None, vmax=None, annot=True, fmt=".2f", figsize=(10, 8)):
        """
        Create a heatmap visualization for correlation or similarity matrices
        
        Parameters:
        -----------
        data_df : pandas DataFrame
            DataFrame containing the data to visualize
        title : str
            Title for the heatmap
        output_file : str
            Path to save the output PDF file
        output_png : str
            Path to save the output PNG file
        cmap : str
            Colormap for the heatmap
        vmin, vmax : float
            Minimum and maximum values for colormap scaling
        annot : bool
            Whether to annotate the heatmap with values
        fmt : str
            Format string for annotations
        figsize : tuple
            Figure size in inches
            
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        plt.figure(figsize=figsize)
        
        # Create mask for diagonal if it's a correlation/similarity matrix
        mask = None
        if data_df.index.equals(data_df.columns):
            mask = np.zeros_like(data_df)
            mask[np.eye(len(data_df), dtype=bool)] = True
        
        # Create the heatmap
        ax = sns.heatmap(data_df, annot=annot, fmt=fmt, cmap=cmap, 
                         mask=mask, vmin=vmin, vmax=vmax, 
                         cbar_kws={'label': 'Correlation' if 'orrelation' in title else 'Similarity'})
        
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        
        # Save the figure if output paths are provided
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
    
    def create_regulon_overlap_chord(self, output_file="shared_reg.pdf", output_png="shared_reg.png"):
        """Create chord diagram showing regulon overlap between subclusters"""
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
        
        # Create chord diagram for regulon overlap
        plt.figure(figsize=(10, 10))
        circos = Circos.chord_diagram(
            overlap_df,
            space=2,
            cmap="tab20",
            r_lim=(85, 100),
            label_kws=dict(r=90, size=10, color="black"),
            link_kws=dict(ec="black", lw=0.3, alpha=0.7),
        )
        plt.title("Regulon Overlap Between Subclusters", pad=20, fontsize=14)
        fig = circos.plotfig()
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
        if output_png:
            plt.savefig(output_png, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create heatmaps for overlap and correlation matrices
        heatmap_dir = os.path.dirname(output_file) if output_file else os.getcwd()
        
        # Heatmap for absolute overlap
        overlap_heatmap_file = os.path.join(heatmap_dir, "regulon_overlap_heatmap.pdf")
        overlap_heatmap_png = os.path.join(heatmap_dir, "regulon_overlap_heatmap.png")
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
        jaccard_heatmap_file = os.path.join(heatmap_dir, "regulon_jaccard_heatmap.pdf")
        jaccard_heatmap_png = os.path.join(heatmap_dir, "regulon_jaccard_heatmap.png")
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
        corr_heatmap_file = os.path.join(heatmap_dir, "regulon_correlation_heatmap.pdf")
        corr_heatmap_png = os.path.join(heatmap_dir, "regulon_correlation_heatmap.png")
        self.create_heatmap(
            correlation_df, 
            "Regulon Correlation Between Subclusters (Pearson)",
            output_file=corr_heatmap_file,
            output_png=corr_heatmap_png,
            cmap="coolwarm",
            vmin=-1,
            vmax=1
        )
        
        return overlap_df, proportion_df, correlation_df
    
    def create_total_target_gene_overlap_chord(self, output_file="total_target_overlap.pdf", output_png="total_target_overlap.png"):
        """Create chord diagram showing total target gene overlap between subclusters"""
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
        
        if overlap_matrix.sum() > len(subcluster_names):
            # Create chord diagram for target gene overlap
            plt.figure(figsize=(10, 10))
            circos = Circos.chord_diagram(
                overlap_df,
                space=2,
                cmap="tab20",
                r_lim=(85, 100),
                label_kws=dict(r=90, size=10, color="black"),
                link_kws=dict(ec="black", lw=0.3, alpha=0.7),
            )
            plt.title("Total Target Gene Overlap Between Subclusters", pad=20, fontsize=14)
            fig = circos.plotfig()
            
            if output_file:
                plt.savefig(output_file, bbox_inches='tight', dpi=300)
            if output_png:
                plt.savefig(output_png, bbox_inches='tight', dpi=300)
            plt.close()
            
            # Create heatmaps for target gene data
            heatmap_dir = os.path.dirname(output_file) if output_file else os.getcwd()
            
            # Heatmap for absolute overlap
            overlap_heatmap_file = os.path.join(heatmap_dir, "target_gene_overlap_heatmap.pdf")
            overlap_heatmap_png = os.path.join(heatmap_dir, "target_gene_overlap_heatmap.png")
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
            jaccard_heatmap_file = os.path.join(heatmap_dir, "target_gene_jaccard_heatmap.pdf")
            jaccard_heatmap_png = os.path.join(heatmap_dir, "target_gene_jaccard_heatmap.png")
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
            corr_heatmap_file = os.path.join(heatmap_dir, "target_gene_correlation_heatmap.pdf")
            corr_heatmap_png = os.path.join(heatmap_dir, "target_gene_correlation_heatmap.png")
            self.create_heatmap(
                correlation_df, 
                "Target Gene Correlation Between Subclusters (Pearson)",
                output_file=corr_heatmap_file,
                output_png=corr_heatmap_png,
                cmap="coolwarm",
                vmin=-1,
                vmax=1
            )
        else:
            print("\nWarning: No significant target gene overlap found between subclusters.")
        
        summary_data = []
        for i, sc1 in enumerate(subcluster_names):
            row = {'Subcluster': sc1, 'Total_Target_Genes': len(all_subcluster_targets[sc1])}
            for j, sc2 in enumerate(subcluster_names):
                if i != j:
                    row[f'Shared_with_{sc2}'] = overlap_matrix[i, j]
                    row[f'Jaccard_with_{sc2}'] = proportional_matrix[i, j]
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        return overlap_df, proportion_df, correlation_df, summary_df, all_subcluster_targets
    
def create_cross_sample_regulon_comparison(self, sample_ids, dataset_id, output_base_dir):
    """Compare regulons across multiple samples with organized output structure and visualize with heatmaps"""
    sample_regulons = defaultdict(set)
    base_scenic_dir = "/work/project/ladcol_020/scGRNi/RNA/SCENIC"
    
    for sample_id in sample_ids:
        sample_path = os.path.join(base_scenic_dir, dataset_id, sample_id)
        if not os.path.exists(sample_path):
            print(f"Warning: Sample directory not found - {sample_path}")
            continue
            
        # Get all regulons from all subclusters for this sample
        for item in os.listdir(sample_path):
            if item.startswith('Tumor_s') and os.path.isdir(os.path.join(sample_path, item)):
                loom_path = os.path.join(sample_path, item, f"{item}_pyscenic_output.loom")
                if os.path.exists(loom_path):
                    with lp.connect(loom_path, mode='r', validate=False) as lf:
                        regulon_names = lf.ca['RegulonsAUC'].dtype.names
                        sample_regulons[sample_id].update(regulon_names)
    
    samples = sorted(sample_regulons.keys())
    
    # Calculate absolute overlap matrix
    overlap_matrix = np.zeros((len(samples), len(samples)), dtype=int)
    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            if i <= j:
                intersection = len(sample_regulons[s1] & sample_regulons[s2])
                overlap_matrix[i,j] = intersection
                overlap_matrix[j,i] = intersection
    
    overlap_df = pd.DataFrame(overlap_matrix,
                             index=samples,
                             columns=samples)
    
    # Calculate Jaccard similarity matrix
    similarity_matrix = np.zeros((len(samples), len(samples)))
    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            if i <= j:
                intersection = len(sample_regulons[s1] & sample_regulons[s2])
                union = len(sample_regulons[s1] | sample_regulons[s2])
                similarity = intersection / union if union > 0 else 0
                similarity_matrix[i,j] = similarity
                similarity_matrix[j,i] = similarity
    
    similarity_df = pd.DataFrame(similarity_matrix,
                               index=samples,
                               columns=samples)
    
    # Calculate Pearson correlation matrix
    correlation_matrix = np.zeros((len(samples), len(samples)))
    all_regulons = set()
    for reg_set in sample_regulons.values():
        all_regulons.update(reg_set)
    all_regulons = sorted(all_regulons)
    
    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            vec1 = np.array([1 if reg in sample_regulons[s1] else 0 for reg in all_regulons])
            vec2 = np.array([1 if reg in sample_regulons[s2] else 0 for reg in all_regulons])
            
            if i == j:
                correlation_matrix[i,j] = 1.0
            else:
                correlation_matrix[i,j] = np.corrcoef(vec1, vec2)[0,1]
    
    correlation_df = pd.DataFrame(correlation_matrix,
                                index=samples,
                                columns=samples)
    
    print("\nAbsolute Regulon Overlap Matrix Between Samples:")
    print(overlap_df)
    print("\nJaccard Similarity Matrix Between Samples:")
    print(similarity_df)
    print("\nPearson Correlation Matrix Between Samples:")
    print(correlation_df)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Save all results to the specified directory
    output_prefix = os.path.join(output_base_dir, "cross_sample_regulon_comparison")
    
    # Save dataframes
    overlap_df.to_csv(f"{output_prefix}_absolute.csv")
    similarity_df.to_csv(f"{output_prefix}_jaccard.csv")
    correlation_df.to_csv(f"{output_prefix}_correlation.csv")
    
    # Create visualizations - one per metric
    # 1. Absolute overlap heatmap
    self.create_heatmap(
        overlap_df,
        "Regulon Overlap Between Samples (Absolute Counts)",
        output_file=f"{output_prefix}_absolute_heatmap.pdf",
        output_png=f"{output_prefix}_absolute_heatmap.png",
        cmap="YlOrRd",
        annot=True,
        fmt="d",
        figsize=(12,10)
    )
    
    # 2. Jaccard similarity heatmap
    self.create_heatmap(
        similarity_df,
        "Regulon Similarity Between Samples (Jaccard Index)",
        output_file=f"{output_prefix}_jaccard_heatmap.pdf",
        output_png=f"{output_prefix}_jaccard_heatmap.png",
        cmap="viridis",
        vmin=0,
        vmax=1,
        figsize=(12,10)
    )
    
    # 3. Pearson correlation heatmap
    self.create_heatmap(
        correlation_df,
        "Regulon Correlation Between Samples (Pearson)",
        output_file=f"{output_prefix}_correlation_heatmap.pdf",
        output_png=f"{output_prefix}_correlation_heatmap.png",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        figsize=(12,10)
    )
    
    print(f"\nSaved all cross-sample comparison results to: {output_prefix}.*")
    
    return overlap_df, similarity_df, correlation_df


def main():
    """Run regulon analysis with organized output structure"""
    dataset_id = "ccRCC_GBM"
    base_samples_dir = "/work/project/ladcol_020/scGRNi/RNA/SCENIC"
    output_base_dir = "/work/project/ladcol_020/integration_GRN_CNV/file_preparation"  # Updated output directory
    
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
    
    first_sample = sample_ids[0]
    base_dir = os.path.join(base_samples_dir, dataset_id)
    analyzer = RegulonAnalyzer(os.path.join(base_dir, first_sample))
    
    print(f"\n{'='*80}")
    print(f"ANALYZING SAMPLE: {first_sample}")
    print(f"{'='*80}")
    
    sample_output_dir = os.path.join(output_base_dir, dataset_id, first_sample)
    os.makedirs(sample_output_dir, exist_ok=True)
    
    regulon_overlap_file = os.path.join(sample_output_dir, f"{first_sample}_regulon_overlap_chord.pdf")
    analyzer.create_regulon_overlap_chord(output_file=regulon_overlap_file)
    
    target_gene_file = os.path.join(sample_output_dir, f"{first_sample}_total_target_gene_overlap_chord.pdf")
    analyzer.create_total_target_gene_overlap_chord(output_file=target_gene_file)
    
    print(f"\n{'='*80}")
    print("COMPARING REGULONS ACROSS SAMPLES")
    print(f"{'='*80}")
    
    # Compare regulons across samples
    overlap_df, similarity_df, correlation_df = analyzer.create_cross_sample_regulon_comparison(
        sample_ids,
        dataset_id,
        os.path.join(output_base_dir, dataset_id)  # Save in dataset-specific subdirectory
    )
    
    print("\n[Complete] All analyses finished with outputs organized by sample")

if __name__ == "__main__":
    main()