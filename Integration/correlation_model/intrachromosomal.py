import pandas as pd
import numpy as np
import os
import pyreadr
from scipy.stats import pearsonr
import warnings
import json
from itertools import combinations
warnings.filterwarnings('ignore')

class IntrachromosomalCorrelationAnalyzer:
    def __init__(self, base_dir="/work/project/ladcol_020"):
        """Initialize with base directory path."""
        self.base_dir = base_dir
        self.gene_order = None

    def load_gene_order(self, gene_order_file):
        """Load gene order file."""
        self.gene_order = pd.read_csv(gene_order_file, sep='\t', header=None,
                                names=['gene', 'chrom', 'start', 'end'])
        self.gene_order['length'] = self.gene_order['end'] - self.gene_order['start']
        self.gene_order.set_index('gene', inplace=True)
        return self.gene_order

    def load_regions_metadata(self, metadata_file):
        """Load regions metadata from JSON file."""
        with open(metadata_file, 'r') as f:
            regions = json.load(f)
        print(f"Loaded {len(regions)} regions from metadata")
        return regions

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

    def group_regions_by_chromosome_and_cnv_class(self, regions):
        """Group regions by chromosome and CNV class for intrachromosomal analysis."""
        chrom_cnv_regions = {}
        
        for region in regions:
            chrom = region['chromosome']
            cnv_class = region['cnv_class']
            key = (chrom, cnv_class)
            
            if key not in chrom_cnv_regions:
                chrom_cnv_regions[key] = []
            chrom_cnv_regions[key].append(region)
        
        # Filter chromosome-CNV combinations with at least 2 regions
        chrom_cnv_regions = {k: v for k, v in chrom_cnv_regions.items() if len(v) >= 2}
        
        print(f"Found regions on {len(chrom_cnv_regions)} chromosome-CNV class combinations suitable for intrachromosomal analysis:")
        for (chrom, cnv_class), regions_list in chrom_cnv_regions.items():
            print(f"  {chrom} ({cnv_class}): {len(regions_list)} regions")
        
        return chrom_cnv_regions

    def calculate_cross_region_correlations(self, expr_data_1, expr_data_2):
        """Calculate correlations between genes from two different regions."""
        correlations = []
        zero_var_genes = set()
        
        genes_1 = expr_data_1.index.tolist()
        genes_2 = expr_data_2.index.tolist()
        
        # Check for zero variance genes
        for gene in genes_1:
            if np.std(expr_data_1.loc[gene].values) == 0:
                zero_var_genes.add(gene)
        
        for gene in genes_2:
            if np.std(expr_data_2.loc[gene].values) == 0:
                zero_var_genes.add(gene)
        
        # Calculate correlations between all gene pairs across regions
        for gene1 in genes_1:
            if gene1 in zero_var_genes:
                continue
                
            x = expr_data_1.loc[gene1].values
            
            for gene2 in genes_2:
                if gene2 in zero_var_genes:
                    continue
                
                y = expr_data_2.loc[gene2].values
                
                try:
                    corr, _ = pearsonr(x, y)
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    continue
        
        total_genes = len(genes_1) + len(genes_2)
        zero_var_pct = len(zero_var_genes) / total_genes * 100 if total_genes > 0 else 0
        
        return correlations, zero_var_pct

    def process_intrachromosomal_correlations(self, expr_matrix, subcluster_df, chrom_cnv_regions, output_file):
        """Process intrachromosomal correlations between regions on the same chromosome with the same CNV class."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            # Write header
            f.write("Chromosome\tCNV_Class\tRegion1_ID\tRegion2_ID\t"
                   "Region1_Start\tRegion1_End\tRegion1_Subtype\t"
                   "Region2_Start\tRegion2_End\tRegion2_Subtype\t"
                   "Distance_bp\tRegion1_Genes\tRegion2_Genes\tGene_Pairs\t"
                   "Mean_Correlation_Altered\tMean_Correlation_Neutral\t"
                   "Median_Correlation_Altered\tMedian_Correlation_Neutral\t"
                   "Cells_Altered\tCells_Neutral\tZero_Var_Genes_Pct\t"
                   "Same_Subtype\n")
            
            total_comparisons = 0
            processed_comparisons = 0
            
            # Process each chromosome-CNV class combination
            for (chrom, cnv_class), regions_list in chrom_cnv_regions.items():
                print(f"\nProcessing chromosome {chrom} with CNV class '{cnv_class}' ({len(regions_list)} regions)")
                
                # Get all pairs of regions on this chromosome with the same CNV class
                region_pairs = list(combinations(regions_list, 2))
                total_comparisons += len(region_pairs)
                
                for region1, region2 in region_pairs:
                    # Get genes for each region
                    genes1 = region1['gene_names']
                    genes2 = region2['gene_names']
                    
                    # Filter to genes that exist in expression matrix
                    common_genes1 = list(set(genes1) & set(expr_matrix.index))
                    common_genes2 = list(set(genes2) & set(expr_matrix.index))
                    
                    if len(common_genes1) < 1 or len(common_genes2) < 1:
                        continue
                    
                    # Get subtypes and neutral subtypes for each region
                    subtype1 = region1['cnv_subtype']
                    subtype2 = region2['cnv_subtype']
                    neutral_subtypes1 = region1['neutral_subtypes']
                    neutral_subtypes2 = region2['neutral_subtypes']
                    
                    # Determine which cells to use for altered and neutral groups
                    # For altered: use cells that are altered in EITHER region
                    altered_cells = set()
                    altered_cells.update(subcluster_df[subcluster_df['subtype'] == subtype1]['cell'])
                    altered_cells.update(subcluster_df[subcluster_df['subtype'] == subtype2]['cell'])
                    altered_cells = list(altered_cells & set(expr_matrix.columns))
                    
                    # For neutral: use cells that are neutral in BOTH regions
                    neutral_cells = set(subcluster_df[subcluster_df['subtype'].isin(neutral_subtypes1)]['cell'])
                    neutral_cells &= set(subcluster_df[subcluster_df['subtype'].isin(neutral_subtypes2)]['cell'])
                    neutral_cells = list(neutral_cells & set(expr_matrix.columns))
                    
                    if len(altered_cells) < 2 or len(neutral_cells) < 2:
                        continue
                    
                    # Get expression data for altered cells
                    expr1_altered = expr_matrix.loc[common_genes1, altered_cells]
                    expr2_altered = expr_matrix.loc[common_genes2, altered_cells]
                    
                    # Get expression data for neutral cells
                    expr1_neutral = expr_matrix.loc[common_genes1, neutral_cells]
                    expr2_neutral = expr_matrix.loc[common_genes2, neutral_cells]
                    
                    # Calculate cross-region correlations
                    corr_altered, zero_var_pct1 = self.calculate_cross_region_correlations(
                        expr1_altered, expr2_altered)
                    corr_neutral, zero_var_pct2 = self.calculate_cross_region_correlations(
                        expr1_neutral, expr2_neutral)
                    
                    if len(corr_altered) == 0:
                        continue
                    
                    # Calculate statistics
                    mean_corr_altered = np.mean(corr_altered)
                    mean_corr_neutral = np.mean(corr_neutral) if corr_neutral else np.nan
                    median_corr_altered = np.median(corr_altered)
                    median_corr_neutral = np.median(corr_neutral) if corr_neutral else np.nan
                    zero_var_pct = max(zero_var_pct1, zero_var_pct2)
                    
                    # Calculate distance between regions
                    distance = abs(region2['start_pos'] - region1['end_pos']) if region2['start_pos'] > region1['end_pos'] else abs(region1['start_pos'] - region2['end_pos'])
                    
                    # Check if regions have same subtype
                    same_subtype = subtype1 == subtype2
                    
                    # Write results
                    f.write(f"{chrom}\t{cnv_class}\t{region1['region_id']}\t{region2['region_id']}\t"
                           f"{region1['start_pos']}\t{region1['end_pos']}\t{subtype1}\t"
                           f"{region2['start_pos']}\t{region2['end_pos']}\t{subtype2}\t"
                           f"{distance}\t{len(common_genes1)}\t{len(common_genes2)}\t{len(corr_altered)}\t"
                           f"{mean_corr_altered:.4f}\t{mean_corr_neutral:.4f}\t"
                           f"{median_corr_altered:.4f}\t{median_corr_neutral:.4f}\t"
                           f"{len(altered_cells)}\t{len(neutral_cells)}\t{zero_var_pct:.1f}\t"
                           f"{same_subtype}\n")
                    
                    processed_comparisons += 1
                    
                    if processed_comparisons % 10 == 0:
                        print(f"  Processed {processed_comparisons}/{total_comparisons} region pairs")
            
            print(f"\nProcessed {processed_comparisons}/{total_comparisons} total region pairs")

    def create_chromosome_summary(self, chrom_cnv_regions, output_file):
        """Create a summary of regions per chromosome and CNV class."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("Chromosome\tCNV_Class\tNum_Regions\tRegion_IDs\tSubtypes\tTotal_Genes\n")
            
            for (chrom, cnv_class), regions_list in chrom_cnv_regions.items():
                region_ids = [str(r['region_id']) for r in regions_list]
                subtypes = list(set(r['cnv_subtype'] for r in regions_list))
                total_genes = sum(r['num_genes'] for r in regions_list)
                
                f.write(f"{chrom}\t{cnv_class}\t{len(regions_list)}\t{','.join(region_ids)}\t"
                       f"{','.join(subtypes)}\t{total_genes}\n")

def main():
    # Initialize analyzer
    base_dir = "/work/project/ladcol_020"
    analyzer = IntrachromosomalCorrelationAnalyzer(base_dir=base_dir)

    dataset_id = "ccRCC_GBM"  
    sample_id = ["C3L-00448-T1_CPT0010160004"]
    
    # Load gene order file
    gene_order_file = os.path.join(base_dir, "scCNV", "inferCNV", "hg38_gencode_v27.txt")
    if not os.path.exists(gene_order_file):
        raise FileNotFoundError(f"Gene order file not found: {gene_order_file}")
    
    analyzer.load_gene_order(gene_order_file)
    print(f"Loaded gene order information for {len(analyzer.gene_order)} genes")
    
    # Load regions metadata
    metadata_file = os.path.join(base_dir, "integration_GRN_CNV/correlation_model", "regions_metadata.json")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Regions metadata file not found: {metadata_file}")
    
    regions = analyzer.load_regions_metadata(metadata_file)
    
    # Group regions by sample and process each sample
    samples = {}
    for region in regions:
        sample_key = (region['dataset_id'], region['sample_id'])
        if sample_key not in samples:
            samples[sample_key] = []
        samples[sample_key].append(region)
    
    print(f"Found regions for {len(samples)} samples")
    
    # Process each sample
    for (dataset_id, sample_id), sample_regions in samples.items():
        print(f"\n{'='*60}")
        print(f"Analyzing intrachromosomal correlations for sample: {sample_id}")
        print(f"{'='*60}")
        
        # Load subcluster information
        subcluster_df = analyzer.load_tumor_subclusters(dataset_id, sample_id)
        if subcluster_df is None:
            continue
        
        # Load expression matrix
        expr_matrix = analyzer.load_expression_matrix(dataset_id, sample_id)
        if expr_matrix is None:
            continue
        
        # Group regions by chromosome and CNV class
        chrom_cnv_regions = analyzer.group_regions_by_chromosome_and_cnv_class(sample_regions)
        
        if not chrom_cnv_regions:
            print(f"No chromosome-CNV class combinations with multiple regions found for sample {sample_id}")
            continue
        
        # Process intrachromosomal correlations
        output_file = os.path.join(base_dir, "integration_GRN_CNV/correlation_model", 
                                 f"{sample_id}_intrachromosomal_correlations_same_cnv.tsv")
        analyzer.process_intrachromosomal_correlations(expr_matrix, subcluster_df, chrom_cnv_regions, output_file)
        
        # Create chromosome summary
        summary_file = os.path.join(base_dir, "integration_GRN_CNV/correlation_model", 
                                  f"{sample_id}_chromosome_cnv_summary.tsv")
        analyzer.create_chromosome_summary(chrom_cnv_regions, summary_file)
        
        print(f"Results saved to {output_file}")
        print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main()