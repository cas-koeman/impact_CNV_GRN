import loompy as lp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from collections import defaultdict

class NonTumorRegulonAnalyzer:
    def __init__(self, base_dir):
        """
        Initialize the analyzer with the base directory containing loom files
        
        Parameters:
        -----------
        base_dir : str
            Base directory containing sample directories with loom files
        """
        self.base_dir = base_dir
        self.sample_data = {}
        
    def find_loom_files(self, sample_pattern=None):
        """
        Find all loom files in the base directory and its subdirectories
        
        Parameters:
        -----------
        sample_pattern : str, optional
            Pattern to filter sample directories
            
        Returns:
        --------
        dict : Dictionary of sample IDs and their loom file paths
        """
        loom_files = {}
        
        # Walk through directory structure
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.loom'):
                    # Extract sample ID from path
                    rel_path = os.path.relpath(root, self.base_dir)
                    if sample_pattern is None or sample_pattern in rel_path:
                        sample_id = rel_path.split(os.sep)[0]  # Assuming sample ID is the first directory level
                        loom_files[sample_id] = os.path.join(root, file)
        
        print(f"Found {len(loom_files)} loom files")
        return loom_files
        
    def load_data(self, loom_paths=None):
        """
        Load regulon AUC from multiple loom files
        
        Parameters:
        -----------
        loom_paths : dict, optional
            Dictionary of sample IDs and their loom file paths
            
        Returns:
        --------
        dict : Dictionary containing loaded data for each sample
        """
        if loom_paths is None:
            loom_paths = self.find_loom_files()
        
        for sample_id, loom_path in loom_paths.items():
            print(f"\nLoading data from {sample_id} ({loom_path})")
            
            try:
                with lp.connect(loom_path, mode='r', validate=False) as lf:
                    # Load regulon AUC matrix if available
                    if 'RegulonsAUC' in lf.ca:
                        # Get regulon names and create DataFrame
                        regulon_auc = pd.DataFrame(lf.ca.RegulonsAUC)
                        regulon_auc.columns = lf.ca.RegulonsAUC.dtype.names
                        
                        # Calculate mean AUC values across all cells in the sample
                        mean_activity = regulon_auc.mean()
                        
                        print(f"  Loaded regulon AUC matrix with shape: {regulon_auc.shape}")
                        print(f"  Calculated mean activity for {len(mean_activity)} regulons")
                        
                        # Store data
                        self.sample_data[sample_id] = {
                            'mean_activity': mean_activity,
                            'regulon_auc': regulon_auc,
                            'loom_path': loom_path,
                            'cell_count': regulon_auc.shape[0]
                        }
                    else:
                        print("  Warning: 'RegulonsAUC' not found in loom file")
            except Exception as e:
                print(f"  Error loading {loom_path}: {str(e)}")
        
        print(f"\nLoaded data for {len(self.sample_data)} samples")
        
        return self.sample_data
    
    def calculate_correlations(self):
        """
        Calculate correlation matrix for mean regulon activity across samples
        
        Returns:
        --------
        pd.DataFrame : Correlation matrix for mean regulon activity
        """
        if not self.sample_data:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        if len(self.sample_data) < 2:
            print("  Warning: At least 2 samples are needed to calculate correlations")
            return None
        
        # Create a DataFrame with mean regulon activity for each sample
        sample_ids = list(self.sample_data.keys())
        regulon_means = pd.DataFrame({
            sample_id: self.sample_data[sample_id]['mean_activity'] 
            for sample_id in sample_ids
        })
        
        # Handle NaN values by filling with 0
        regulon_means = regulon_means.fillna(0)
        
        # Calculate correlation matrix
        correlation_matrix = regulon_means.corr(method='pearson')
        
        print("\nCorrelation matrix for non-tumor cells:")
        print(correlation_matrix)
        
        return correlation_matrix
    
    def calculate_regulon_overlap(self):
        """
        Calculate regulon overlap and Jaccard similarity across samples
        
        Returns:
        --------
        tuple : (overlap_df, similarity_df) containing overlap and Jaccard matrices
        """
        if not self.sample_data:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        sample_ids = list(self.sample_data.keys())
        
        if len(sample_ids) < 2:
            print("  Warning: At least 2 samples are needed to calculate overlap")
            return None, None
        
        # For each sample, get the active regulons (mean AUC > 0)
        sample_regulons = {}
        for sample_id in sample_ids:
            # Get mean regulon activity for this sample
            regulon_activity = self.sample_data[sample_id]['mean_activity']
            # Consider regulons with activity > 0 as "active"
            active_regulons = set(regulon_activity[regulon_activity > 0].index)
            sample_regulons[sample_id] = active_regulons
            print(f"  Sample {sample_id}: {len(active_regulons)} active regulons")
        
        # Create overlap matrix
        overlap_matrix = np.zeros((len(sample_ids), len(sample_ids)), dtype=int)
        similarity_matrix = np.zeros((len(sample_ids), len(sample_ids)))
        
        for i, s1 in enumerate(sample_ids):
            for j, s2 in enumerate(sample_ids):
                if i <= j:  # Only calculate upper triangle
                    intersection = len(sample_regulons[s1] & sample_regulons[s2])
                    union = len(sample_regulons[s1] | sample_regulons[s2])
                    
                    # Absolute overlap
                    overlap_matrix[i, j] = intersection
                    overlap_matrix[j, i] = intersection
                    
                    # Jaccard similarity
                    similarity = intersection / union if union > 0 else 0
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        overlap_df = pd.DataFrame(overlap_matrix, index=sample_ids, columns=sample_ids)
        similarity_df = pd.DataFrame(similarity_matrix, index=sample_ids, columns=sample_ids)
        
        print("\nRegulons Absolute Overlap Matrix:")
        print(overlap_df)
        
        print("\nRegulons Jaccard Similarity Matrix:")
        print(similarity_df)
        
        return overlap_df, similarity_df
    
    def plot_correlation_heatmap(self, correlation_df, output_file=None):
        """
        Plot correlation heatmap
        
        Parameters:
        -----------
        correlation_df : pd.DataFrame
            Correlation matrix
        output_file : str, optional
            Base output file path without extension
            
        Returns:
        --------
        None
        """
        if correlation_df is None:
            print("No correlation matrix available to plot")
            return
        
        # Lower-triangle heatmap for Pearson correlation
        mask = np.triu(np.ones_like(correlation_df, dtype=bool), k=1)
        fig, ax = plt.subplots(figsize=(10, 9))
        sns.heatmap(correlation_df, mask=mask, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
                    cbar_kws={"orientation": "horizontal", "shrink": 0.6, "pad": 0.25, "label": "Pearson coefficient"}, ax=ax)
        plt.title("Non-Tumor Regulon Correlation Between Samples", fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(f"{output_file}_correlation_heatmap.pdf", bbox_inches='tight', dpi=300)
            plt.savefig(f"{output_file}_correlation_heatmap.png", bbox_inches='tight', dpi=300)
            print(f"Saved correlation heatmap to {output_file}_correlation_heatmap.pdf/png")
        
        plt.close()
    
    def plot_overlap_heatmap(self, overlap_df, output_file=None):
        """
        Plot absolute overlap heatmap
        
        Parameters:
        -----------
        overlap_df : pd.DataFrame
            Overlap matrix
        output_file : str, optional
            Base output file path without extension
            
        Returns:
        --------
        None
        """
        if overlap_df is None:
            print("No overlap matrix available to plot")
            return
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(overlap_df, annot=True, fmt="d", cmap="YlOrRd")
        plt.title("Non-Tumor Regulon Overlap Between Samples (Absolute Counts)", fontsize=14, pad=20)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(f"{output_file}_absolute_heatmap.pdf", bbox_inches='tight', dpi=300)
            plt.savefig(f"{output_file}_absolute_heatmap.png", bbox_inches='tight', dpi=300)
            print(f"Saved absolute overlap heatmap to {output_file}_absolute_heatmap.pdf/png")
        
        plt.close()
    
    def plot_jaccard_heatmap(self, similarity_df, output_file=None):
        """
        Plot Jaccard similarity heatmap
        
        Parameters:
        -----------
        similarity_df : pd.DataFrame
            Jaccard similarity matrix
        output_file : str, optional
            Base output file path without extension
            
        Returns:
        --------
        None
        """
        if similarity_df is None:
            print("No similarity matrix available to plot")
            return
        
        # Lower-triangle heatmap for Jaccard similarity
        mask = np.triu(np.ones_like(similarity_df, dtype=bool), k=1)
        fig, ax = plt.subplots(figsize=(10, 9))
        sns.heatmap(similarity_df, mask=mask, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
                    cbar_kws={"orientation": "horizontal", "shrink": 0.6, "pad": 0.25, "label": "Jaccard index"}, ax=ax)
        plt.title("Non-Tumor Regulon Similarity Between Samples", fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(f"{output_file}_jaccard_heatmap.pdf", bbox_inches='tight', dpi=300)
            plt.savefig(f"{output_file}_jaccard_heatmap.png", bbox_inches='tight', dpi=300)
            print(f"Saved Jaccard similarity heatmap to {output_file}_jaccard_heatmap.pdf/png")
        
        plt.close()
    
    def find_common_regulons(self, threshold=0.75):
        """
        Find regulons that are active across multiple samples
        
        Parameters:
        -----------
        threshold : float
            Proportion of samples a regulon must be active in to be considered common
            
        Returns:
        --------
        pd.DataFrame : DataFrame of common regulons with their mean activity across samples
        """
        if not self.sample_data:
            raise ValueError("No data loaded. Please call load_data() first.")
            
        sample_ids = list(self.sample_data.keys())
        regulon_counts = defaultdict(int)
        regulon_activities = defaultdict(list)
        
        # Count how many samples each regulon is active in
        for sample_id in sample_ids:
            mean_activity = self.sample_data[sample_id]['mean_activity']
            for regulon, activity in mean_activity.items():
                if activity > 0:
                    regulon_counts[regulon] += 1
                    regulon_activities[regulon].append(activity)
        
        # Calculate the minimum samples required based on threshold
        min_samples = max(2, int(len(sample_ids) * threshold))
        
        # Find common regulons
        common_regulons = {}
        for regulon, count in regulon_counts.items():
            if count >= min_samples:
                common_regulons[regulon] = {
                    'sample_count': count,
                    'sample_percentage': count / len(sample_ids) * 100,
                    'mean_activity': np.mean(regulon_activities[regulon])
                }
        
        # Create DataFrame
        common_df = pd.DataFrame.from_dict(common_regulons, orient='index')
        common_df = common_df.sort_values('sample_percentage', ascending=False)
        
        print(f"\nFound {len(common_df)} regulons active in at least {min_samples} samples ({threshold*100:.1f}%)")
        print(common_df.head(20))
        
        return common_df
    
    def run_analysis(self, output_dir=None):
        """
        Run full analysis pipeline for non-tumor regulons
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save output files
            
        Returns:
        --------
        dict : Dictionary containing analysis results
        """
        print(f"\n{'='*80}")
        print(f"ANALYZING NON-TUMOR REGULONS ACROSS SAMPLES")
        print(f"{'='*80}")
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output will be saved to: {output_dir}")
        
        # Calculate correlations
        correlation_matrix = self.calculate_correlations()
        
        # Calculate overlap and Jaccard similarity
        overlap_df, similarity_df = self.calculate_regulon_overlap()
        
        # Find common regulons
        common_regulons = self.find_common_regulons(threshold=0.75)
        
        if correlation_matrix is None:
            print("Could not analyze - not enough data")
            return None
        
        results = {
            'correlation_matrix': correlation_matrix,
            'overlap_matrix': overlap_df,
            'similarity_matrix': similarity_df,
            'common_regulons': common_regulons
        }
        
        # Create base output file path
        if output_dir:
            base_output_file = os.path.join(output_dir, "non_tumor_regulons")
        else:
            base_output_file = None
        
        # Generate all visualizations
        self.plot_correlation_heatmap(correlation_matrix, base_output_file)
        self.plot_overlap_heatmap(overlap_df, base_output_file)
        self.plot_jaccard_heatmap(similarity_df, base_output_file)
        
        # Save matrices to CSV
        if output_dir:
            correlation_matrix.to_csv(f"{base_output_file}_correlation_matrix.csv")
            overlap_df.to_csv(f"{base_output_file}_overlap_matrix.csv")
            similarity_df.to_csv(f"{base_output_file}_similarity_matrix.csv")
            common_regulons.to_csv(f"{base_output_file}_common_regulons.csv")
            print(f"Saved all matrices to CSV files in {output_dir}")
        
        return results


# If this script is run directly
if __name__ == "__main__":
    # Base directory containing sample folders with loom files
    base_dir = "/work/project/ladcol_020/scGRNi/RNA/SCENIC/ccRCC_GBM"
    
    # List of sample IDs to analyze
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
    
    # Output directory for results
    output_dir = "/work/project/ladcol_020/integration_GRN_CNV/scenic_analysis/non_tumor_comp"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"STARTING NON-TUMOR REGULON ANALYSIS")
    print(f"{'='*80}")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create analyzer
    analyzer = NonTumorRegulonAnalyzer(base_dir)
    
    # Create a dictionary of sample IDs and their loom file paths
    loom_files = {}
    for sample_id in sample_ids:
        loom_path = os.path.join(base_dir, sample_id, "Non-Tumor_pyscenic_output.loom")
        if os.path.exists(loom_path):
            loom_files[sample_id] = loom_path
    
    print(f"\nFound {len(loom_files)} loom files from specified sample IDs")
    for sample_id, loom_path in loom_files.items():
        print(f"- {sample_id}: {loom_path}")
    
    # Load data from all loom files
    analyzer.load_data(loom_files)
    
    # Run analysis
    analyzer.run_analysis(output_dir)
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE - Results saved to: {output_dir}")
    print(f"{'='*80}")