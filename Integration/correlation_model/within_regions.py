import pandas as pd
import numpy as np
import os
import pyreadr
from scipy.stats import pearsonr
import warnings
import json
warnings.filterwarnings('ignore')

class CNVRegionFinder:
    def __init__(self, base_dir="/work/project/ladcol_020"):
        """Initialize with base directory path."""
        self.base_dir = base_dir
        self.gene_lengths = None
        self.total_genome_length = None
        self.gene_order = None

    def load_tumor_subclusters(self, dataset_id, sample_id):
        """Load tumor subclusters from RDS file."""
        subcluster_path = os.path.join(
            self.base_dir, "scCNV/inferCNV",
            dataset_id, sample_id, "tumor_subclusters.rds"
        )

        if not os.path.exists(subcluster_path):
            print(f"[Warning] Tumor subclusters file not found at {subcluster_path}")
            return None

        try:
            result = pyreadr.read_r(subcluster_path)
            df = result[None]
            df.columns = ['cell', 'subtype']
            return df
        except Exception as e:
            print(f"[Error] Loading tumor subclusters: {str(e)}")
            return None

    def load_gene_order(self, gene_order_file):
        """Load gene order file and prepare gene length dictionary."""
        self.gene_order = pd.read_csv(gene_order_file, sep='\t', header=None,
                                names=['gene', 'chrom', 'start', 'end'])
        self.gene_order['length'] = self.gene_order['end'] - self.gene_order['start']
        self.gene_order.set_index('gene', inplace=True)

        # Create gene length dictionary
        self.gene_lengths = dict(zip(self.gene_order.index, self.gene_order['length']))
        self.total_genome_length = sum(self.gene_lengths.values())

        return self.gene_order

    def load_cnv_matrix(self, cnv_file_path):
        """Load CNV matrix from file."""
        try:
            cnv_matrix = pd.read_csv(cnv_file_path, sep='\t', index_col=0)
            print(f"CNV matrix loaded: {cnv_matrix.shape[0]} genes x {cnv_matrix.shape[1]} cells")
            return cnv_matrix
        except Exception as e:
            print(f"[Error] Loading CNV matrix: {str(e)}")
            return None

    def load_expression_matrix(self, dataset_id, sample_id):
        """Load raw expression matrix for the sample."""
        expr_path = os.path.join(
            self.base_dir, "integration_GRN_CNV", 
            dataset_id, sample_id, "raw_count_matrix.txt"
        )
        
        if not os.path.exists(expr_path):
            print(f"[Warning] Expression matrix not found at {expr_path}")
            return None
            
        try:
            expr_matrix = pd.read_csv(expr_path, sep='\t', index_col=0)
            print(f"Expression matrix loaded: {expr_matrix.shape[0]} genes x {expr_matrix.shape[1]} cells")
            return expr_matrix
        except Exception as e:
            print(f"[Error] Loading expression matrix: {str(e)}")
            return None

    def get_subtype_cnv_averages(self, cnv_matrix, subcluster_df):
        """Calculate average CNV values per subtype for each gene."""
        # Filter cells that exist in both datasets
        common_cells = set(cnv_matrix.columns) & set(subcluster_df['cell'])
        
        if len(common_cells) == 0:
            print("[Warning] No common cells found between CNV matrix and subcluster data")
            return None
            
        print(f"Found {len(common_cells)} common cells")
        
        # Filter datasets to common cells
        cnv_filtered = cnv_matrix[list(common_cells)]
        subcluster_filtered = subcluster_df[subcluster_df['cell'].isin(common_cells)]
        
        # Calculate average CNV per subtype
        subtype_averages = {}
        for subtype in subcluster_filtered['subtype'].unique():
            subtype_cells = subcluster_filtered[subcluster_filtered['subtype'] == subtype]['cell']
            subtype_cnv = cnv_filtered[subtype_cells]
            subtype_averages[subtype] = subtype_cnv.mean(axis=1)
            
        return pd.DataFrame(subtype_averages)

    def add_genomic_positions(self, cnv_df):
        """Add genomic position information to CNV dataframe."""
        # Filter to genes that exist in gene_order
        common_genes = set(cnv_df.index) & set(self.gene_order.index)
        cnv_filtered = cnv_df.loc[list(common_genes)]
        
        # Add genomic information
        cnv_with_pos = cnv_filtered.copy()
        cnv_with_pos['chrom'] = self.gene_order.loc[cnv_filtered.index, 'chrom']
        cnv_with_pos['start'] = self.gene_order.loc[cnv_filtered.index, 'start']
        cnv_with_pos['end'] = self.gene_order.loc[cnv_filtered.index, 'end']
        
        # Sort by chromosome and position
        cnv_with_pos['chrom_num'] = cnv_with_pos['chrom'].str.replace('chr', '').replace('X', '23').replace('Y', '24').astype(int)
        cnv_with_pos = cnv_with_pos.sort_values(['chrom_num', 'start'])
        
        return cnv_with_pos

    def classify_cnv_value(self, value, tolerance=0.05):
        """Classify CNV value into categories with tolerance."""
        if abs(value - 0) < tolerance:
            return 'double_loss'
        elif abs(value - 0.5) < tolerance:
            return 'single_loss'
        elif abs(value - 1) < tolerance:
            return 'neutral'
        elif abs(value - 1.5) < tolerance:
            return 'single_gain'
        elif abs(value - 2) < tolerance:
            return 'double_gain'
        elif value > 2.5:
            return 'multiple_gains'
        else:
            return 'other'

    def find_consecutive_regions(self, cnv_with_pos, min_length=3, tolerance=0.05):
        """Find consecutive regions with same CNV state per subtype."""
        regions_of_interest = []
        subtypes = [col for col in cnv_with_pos.columns if col not in ['chrom', 'start', 'end', 'chrom_num']]
        
        print(f"Analyzing {len(subtypes)} subtypes: {subtypes}")
        
        # Group by chromosome
        for chrom in cnv_with_pos['chrom'].unique():
            chrom_data = cnv_with_pos[cnv_with_pos['chrom'] == chrom].copy()
            
            if len(chrom_data) < min_length:
                continue
                
            print(f"Processing {chrom} with {len(chrom_data)} genes")
            
            # For each subtype, find consecutive regions
            for subtype in subtypes:
                cnv_values = chrom_data[subtype].values
                positions = list(range(len(cnv_values)))
                
                # Classify CNV values
                cnv_classes = [self.classify_cnv_value(val, tolerance) for val in cnv_values]
                
                # Find consecutive regions of same CNV class
                i = 0
                while i < len(cnv_classes):
                    current_class = cnv_classes[i]
                    
                    # Skip neutral regions for now (we'll check them later)
                    if current_class == 'neutral' or current_class == 'other':
                        i += 1
                        continue
                    
                    # Find end of consecutive region
                    j = i
                    while j < len(cnv_classes) and cnv_classes[j] == current_class:
                        j += 1
                    
                    # Check if region is long enough
                    if j - i >= min_length:
                        region_genes = chrom_data.iloc[i:j]
                        region_start = region_genes['start'].min()
                        region_end = region_genes['end'].max()
                        
                        # Check if other subtypes have different CNV in this region
                        is_differential = False
                        other_subtypes_neutral = []
                        
                        for other_subtype in subtypes:
                            if other_subtype == subtype:
                                continue
                                
                            other_cnv_values = chrom_data.iloc[i:j][other_subtype].values
                            other_classes = [self.classify_cnv_value(val, tolerance) for val in other_cnv_values]
                            
                            # Check if majority of other subtype is neutral
                            neutral_count = sum(1 for c in other_classes if c == 'neutral')
                            if neutral_count >= len(other_classes) * 0.95:  # 95% neutral
                                other_subtypes_neutral.append(other_subtype)
                                is_differential = True
                        
                        if is_differential:
                            region_info = {
                                'region_id': len(regions_of_interest) + 1,
                                'chromosome': chrom,
                                'start_pos': region_start,
                                'end_pos': region_end,
                                'length_bp': region_end - region_start,
                                'num_genes': j - i,
                                'cnv_subtype': subtype,
                                'cnv_class': current_class,
                                'cnv_values': cnv_values[i:j].tolist(),
                                'neutral_subtypes': other_subtypes_neutral,
                                'gene_names': region_genes.index.tolist()
                            }
                            regions_of_interest.append(region_info)
                    
                    i = j
        
        return regions_of_interest

    def calculate_pairwise_correlations(self, expr_data):
        """Calculate all pairwise Pearson correlations within a gene set."""
        correlations = []
        zero_var_genes = set()
        
        genes = expr_data.index.tolist()
        
        for i, gene1 in enumerate(genes):
            x = expr_data.loc[gene1].values
            if np.std(x) == 0:
                zero_var_genes.add(gene1)
                continue
                
            for j, gene2 in enumerate(genes):
                if i >= j:  # Skip self-comparisons and duplicates
                    continue
                    
                y = expr_data.loc[gene2].values
                if np.std(y) == 0:
                    zero_var_genes.add(gene2)
                    continue
                
                try:
                    corr, _ = pearsonr(x, y)
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    continue
        
        zero_var_pct = len(zero_var_genes) / len(genes) * 100 if genes else 0
        return correlations, zero_var_pct

    def process_within_region_correlations(self, expr_matrix, subcluster_df, regions, output_file):
        """Process within-region correlations for all regions."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("Region_ID\tChromosome\tStart\tEnd\tSubtype\tNeutral_Subtypes\t"
                   "Gene_Count\tGene_Pairs\tMean_Correlation_Altered\tMean_Correlation_Neutral\t"
                   "Median_Correlation_Altered\tMedian_Correlation_Neutral\t"
                   "Cells_Altered\tCells_Neutral\tZero_Var_Genes_Pct\n")
            
            for region in regions:
                gene_names = region['gene_names']
                subtype = region['cnv_subtype']
                neutral_subtypes = region['neutral_subtypes']
                
                # Filter to genes that exist in expression matrix
                common_genes = list(set(gene_names) & set(expr_matrix.index))
                if len(common_genes) < 2:
                    print(f"Region {region['region_id']} has insufficient genes ({len(common_genes)}) in expression matrix")
                    continue
                    
                # Get cells for each group
                subtype_cells = list(set(subcluster_df[subcluster_df['subtype'] == subtype]['cell']) & 
                                   set(expr_matrix.columns))
                neutral_cells = list(set(subcluster_df[subcluster_df['subtype'].isin(neutral_subtypes)]['cell']) & 
                                   set(expr_matrix.columns))
                
                if len(subtype_cells) < 2 or len(neutral_cells) < 2:
                    print(f"Region {region['region_id']} has insufficient cells (altered: {len(subtype_cells)}, neutral: {len(neutral_cells)})")
                    continue
                
                # Get expression data for this region
                expr_subtype = expr_matrix.loc[common_genes, subtype_cells]
                expr_neutral = expr_matrix.loc[common_genes, neutral_cells]
                
                # Calculate correlations
                corr_subtype, zero_var_pct1 = self.calculate_pairwise_correlations(expr_subtype)
                corr_neutral, zero_var_pct2 = self.calculate_pairwise_correlations(expr_neutral)
                
                # Calculate statistics
                n_pairs = len(corr_subtype)
                if n_pairs == 0:
                    continue
                    
                mean_corr_subtype = np.mean(corr_subtype)
                mean_corr_neutral = np.mean(corr_neutral) if corr_neutral else np.nan
                median_corr_subtype = np.median(corr_subtype)
                median_corr_neutral = np.median(corr_neutral) if corr_neutral else np.nan
                zero_var_pct = max(zero_var_pct1, zero_var_pct2)
                
                # Write to file
                f.write(f"{region['region_id']}\t{region['chromosome']}\t{region['start_pos']}\t{region['end_pos']}\t"
                       f"{subtype}\t{','.join(neutral_subtypes)}\t{len(common_genes)}\t{n_pairs}\t"
                       f"{mean_corr_subtype:.4f}\t{mean_corr_neutral:.4f}\t"
                       f"{median_corr_subtype:.4f}\t{median_corr_neutral:.4f}\t"
                       f"{len(subtype_cells)}\t{len(neutral_cells)}\t{zero_var_pct:.1f}\n")
                
                print(f"Processed region {region['region_id']} with {n_pairs} gene pairs")

    def save_regions_metadata(self, regions, output_file):
        """Save region metadata to JSON file for use by other scripts."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(regions, f, indent=2)
        print(f"Saved {len(regions)} regions metadata to {output_file}")

def main():
    # Initialize analyzer
    base_dir = "/work/project/ladcol_020"
    analyzer = CNVRegionFinder(base_dir=base_dir)
    
    # Load gene order file
    gene_order_file = os.path.join(base_dir, "scCNV", "inferCNV", "hg38_gencode_v27.txt")
    if not os.path.exists(gene_order_file):
        raise FileNotFoundError(f"Gene order file not found: {gene_order_file}")
    
    analyzer.load_gene_order(gene_order_file)
    print(f"Loaded gene order information for {len(analyzer.gene_order)} genes")
    
    # Define your dataset and sample IDs
    dataset_id = "ccRCC_GBM"  
    sample_ids = ["C3L-00448-T1_CPT0010160004"]
    
    all_regions = []
    
    # Process each sample
    for sample_id in sample_ids:
        print(f"\n{'='*60}")
        print(f"Analyzing sample: {sample_id}")
        print(f"{'='*60}")
        
        # Load subcluster information
        subcluster_df = analyzer.load_tumor_subclusters(dataset_id, sample_id)
        if subcluster_df is None:
            continue
        
        print(f"Found {len(subcluster_df)} cells in {len(subcluster_df['subtype'].unique())} subtypes")
        
        # Load CNV matrix
        cnv_file = os.path.join(base_dir, "integration_GRN_CNV", dataset_id, sample_id,
                               "extended_cnv_matrix.tsv")
        
        if not os.path.exists(cnv_file):
            print(f"Warning: CNV file not found: {cnv_file}")
            continue
            
        cnv_matrix = analyzer.load_cnv_matrix(cnv_file)
        if cnv_matrix is None:
            continue
        
        # Load expression matrix
        expr_matrix = analyzer.load_expression_matrix(dataset_id, sample_id)
        if expr_matrix is None:
            continue
        
        # Calculate subtype averages
        subtype_cnv = analyzer.get_subtype_cnv_averages(cnv_matrix, subcluster_df)
        if subtype_cnv is None:
            continue
            
        print(f"Subtype CNV averages calculated for {subtype_cnv.shape[1]} subtypes")
        
        # Add genomic positions
        cnv_with_pos = analyzer.add_genomic_positions(subtype_cnv)
        print(f"Added genomic positions for {len(cnv_with_pos)} genes")
        
        # Find differential regions
        regions = analyzer.find_consecutive_regions(cnv_with_pos, min_length=3)
        print(f"Found {len(regions)} differential regions")
        
        # Add sample ID to each region
        for region in regions:
            region['sample_id'] = sample_id
            region['dataset_id'] = dataset_id
        
        all_regions.extend(regions)
        
        # Process within-region correlations
        output_file = os.path.join(base_dir, "integration_GRN_CNV/correlation_model", 
                                 f"{sample_id}_within_region_correlations.tsv")
        analyzer.process_within_region_correlations(expr_matrix, subcluster_df, regions, output_file)
    
    # Save all regions metadata
    metadata_file = os.path.join(base_dir, "integration_GRN_CNV/correlation_model", "regions_metadata.json")
    analyzer.save_regions_metadata(all_regions, metadata_file)
    
    print(f"\nTotal regions found across all samples: {len(all_regions)}")

if __name__ == "__main__":
    main()