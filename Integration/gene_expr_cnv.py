import os
import numpy as np
import pandas as pd
from scipy import stats


class SingleGeneExpressionCNVAnalyzer:
    def __init__(self, gene_symbol, datasets, cnv_dir, raw_count_dir):
        """
        Initialize the class with a gene symbol, dataset names, and directory paths.

        Args:
            gene_symbol (str): Symbol of the gene to analyze.
            datasets (list): List of dataset names (tumor samples).
            cnv_dir (str): Directory for CNV matrix files.
            raw_count_dir (str): Directory for raw count matrix files.
        """
        self.gene_symbol = gene_symbol
        self.datasets = datasets
        self.cnv_dir = cnv_dir
        self.raw_count_dir = raw_count_dir
        self.results = []
        
    def analyze_gene(self):
        """
        Analyze the specified gene across all datasets.
        Calculate z-score normalized expression and match with CNV data.
        """
        print(f"\nAnalyzing gene {self.gene_symbol} across {len(self.datasets)} tumor samples...")
        
        # Process each dataset
        for dataset in self.datasets:
            try:
                # Define file paths
                cnv_matrix_path = os.path.join(self.cnv_dir, dataset, "extended_cnv_matrix.tsv")
                raw_count_path = os.path.join(self.raw_count_dir, dataset, "raw_count_matrix.txt")
                
                # Check if files exist
                if not os.path.exists(cnv_matrix_path):
                    print(f"Warning: CNV matrix file not found for {dataset}, skipping.")
                    continue
                    
                if not os.path.exists(raw_count_path):
                    print(f"Warning: Raw count matrix file not found for {dataset}, skipping.")
                    continue
                
                # Load matrices
                print(f"Loading data for {dataset}...")
                cnv_matrix = pd.read_csv(cnv_matrix_path, sep='\t', index_col=0)
                raw_count_matrix = pd.read_csv(raw_count_path, sep='\t', index_col=0)
                
                # Convert CNV to copy number change relative to diploid
                cnv_matrix = cnv_matrix 
                
                print(f"CNV matrix dimensions: {cnv_matrix.shape}")
                print(f"Raw count matrix dimensions (before filtering): {raw_count_matrix.shape}")
                
                # Filter genes and cells
                gene_filter = (raw_count_matrix > 0).sum(axis=1) >= 200
                filtered_matrix = raw_count_matrix[gene_filter]
                cell_filter = (filtered_matrix > 0).sum(axis=0) >= 3
                filtered_matrix = filtered_matrix.loc[:, cell_filter]
                
                print(f"Filtered raw count matrix dimensions: {filtered_matrix.shape}")
                
                # Z-normalize expression data
                z_score_matrix = filtered_matrix.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
                print(f"Z-score matrix dimensions: {z_score_matrix.shape}")
                
                # Check if gene exists in both matrices after filtering
                if self.gene_symbol not in cnv_matrix.index:
                    print(f"Warning: Gene {self.gene_symbol} not found in CNV matrix for {dataset}, skipping.")
                    continue
                    
                if self.gene_symbol not in z_score_matrix.index:
                    print(f"Warning: Gene {self.gene_symbol} not found in filtered expression matrix for {dataset}, skipping.")
                    continue
                
                # Extract gene data
                gene_cnv = cnv_matrix.loc[self.gene_symbol]
                gene_z_scores = z_score_matrix.loc[self.gene_symbol]
                
                # Match cells between CNV and expression data
                common_cells = gene_cnv.index.intersection(gene_z_scores.index)
                
                if len(common_cells) == 0:
                    print(f"Warning: No common cells found between CNV and expression data for {dataset}, skipping.")
                    continue
                
                # Filter to common cells
                gene_cnv = gene_cnv[common_cells]
                gene_z_scores = gene_z_scores[common_cells]
                
                # Calculate average z-score for the gene in this dataset
                avg_z_score = np.mean(gene_z_scores)
                
                # Calculate average CNV for the gene in this dataset
                avg_cnv = np.mean(gene_cnv)
                
                # Store results
                self.results.append({
                    'Tumor Sample': dataset,
                    'Avg Z-Score Expression': avg_z_score,
                    'Avg Copy Number': avg_cnv,
                    'Cell Count': len(common_cells)
                })
                
                print(f"Processed {dataset}: Avg Z-Score = {avg_z_score:.4f}, Avg CNV = {avg_cnv:.4f}, Cells = {len(common_cells)}")
                
            except Exception as e:
                print(f"Error processing dataset {dataset}: {str(e)}")
    
    def display_results(self):
        """
        Display the results as a simple table.
        """
        if not self.results:
            print(f"No results found for gene {self.gene_symbol}.")
            return None
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Sort by absolute z-score (to show strongest signals first)
        results_df = results_df.sort_values(by='Avg Z-Score Expression', key=abs, ascending=False)
        
        # Display the table header
        print(f"\nResults for gene {self.gene_symbol}:")
        print(f"{'Tumor Sample':<30} {'Avg Z-Score Expression':<25} {'Avg Copy Number':<20} {'Cell Count':<10}")
        print("-" * 85)
        
        # Display each row
        for _, row in results_df.iterrows():
            print(f"{row['Tumor Sample']:<30} {row['Avg Z-Score Expression']:<25.4f} {row['Avg Copy Number']:<20.4f} {row['Cell Count']:<10}")
        
        return results_df


def main():
    # Get user input for gene symbol
    gene_symbol = input("Enter the gene symbol to analyze (e.g. VHL, TP53): ")
    
    # Sample IDs (tumor samples)
    sample_ids = [
        "C3L-00004-T1_CPT0001540013",
        "C3L-00026-T1_CPT0001500003",
        "C3L-00088-T1_CPT0000870003"
        "C3L-00416-T2_CPT0010100001",
        "C3L-00448-T1_CPT0010160004",
        "C3L-00917-T1_CPT0023690004",
        "C3L-01313-T1_CPT0086820004",
        "C3N-00317-T1_CPT0012280004",
        "C3N-00495-T1_CPT0078510004"
    ]
    
    # Set paths for CNV and expression data
    cnv_dir = "/work/project/ladcol_020/integration_GRN_CNV/ccRCC_GBM/"
    expr_dir = "/work/project/ladcol_020/integration_GRN_CNV/ccRCC_GBM/"
    
    # Initialize analyzer
    analyzer = SingleGeneExpressionCNVAnalyzer(
        gene_symbol=gene_symbol,
        datasets=sample_ids,
        cnv_dir=cnv_dir,
        raw_count_dir=expr_dir
    )
    
    # Run analysis
    analyzer.analyze_gene()
    
    # Display results
    results_df = analyzer.display_results()
    
    # Save results to CSV
    output_file = f"{gene_symbol}_expression_cnv_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()