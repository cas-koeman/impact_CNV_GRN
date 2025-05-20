#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadr  # Make sure to import this for reading RDS files
from scipy.optimize import minimize


class CopyNumberMetrics:
    """Class to analyze CNA burden and heterogeneity across tumor subtypes."""

    def __init__(self, base_dir="/work/project/ladcol_020"):
        """Initialize with base directory path."""
        self.base_dir = base_dir
        self.gene_lengths = None
        self.total_genome_length = None

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
            # Extract the DataFrame from the OrderedDict (it's stored under the 'None' key)
            df = result[None]
            # Rename columns to standard names for easier processing
            df.columns = ['cell', 'subtype']
            return df
        except Exception as e:
            print(f"[Error] Loading tumor subclusters: {str(e)}")
            return None

    def load_gene_order(self, gene_order_file):
        """Load gene order file and prepare gene length dictionary."""
        gene_order = pd.read_csv(gene_order_file, sep='\t', header=None,
                                 names=['gene', 'chrom', 'start', 'end'])
        gene_order['length'] = gene_order['end'] - gene_order['start']

        # Create gene length dictionary
        self.gene_lengths = dict(zip(gene_order['gene'], gene_order['length']))
        self.total_genome_length = sum(self.gene_lengths.values())

        return gene_order

    def calculate_segmented_copy_numbers(self, cnv_df, window_size=100):
        """
        Segment the genome and calculate mean copy numbers for each segment.

        Args:
            cnv_df: DataFrame with copy number data (genes x cells)
            window_size: Number of genes to include in each segment

        Returns:
            Dictionary of segmented copy numbers per cell
        """
        segmented_data = {}

        # Sort genes by genomic position (assuming gene_order has been loaded)
        sorted_genes = sorted(self.gene_lengths.keys(), key=lambda x: (self.gene_order.loc[x, 'chrom'],
                                                                       self.gene_order.loc[x, 'start']))

        for cell in cnv_df.columns:
            cell_segments = []
            current_segment = []

            for i, gene in enumerate(sorted_genes):
                if gene in cnv_df.index:
                    current_segment.append(cnv_df.loc[gene, cell])

                    # When we reach window size or end of genome, process segment
                    if len(current_segment) >= window_size or i == len(sorted_genes) - 1:
                        # Calculate mean CN for segment
                        mean_cn = np.mean(current_segment)
                        segment_width = sum(
                            self.gene_lengths[g] for g in sorted_genes[i - len(current_segment) + 1:i + 1]
                            if g in self.gene_lengths)
                        cell_segments.append({
                            'mean_cn': mean_cn,
                            'width': segment_width
                        })
                        current_segment = []

            segmented_data[cell] = cell_segments

        return segmented_data

    def calculate_cnh(self, segmented_data):
        """
        Calculate Copy Number Heterogeneity (CNH) for each cell.

        Args:
            segmented_data: Dictionary of segmented copy numbers per cell

        Returns:
            Dictionary with CNH values per cell
        """
        cnh_values = {}

        for cell, segments in segmented_data.items():
            # Extract segment means and widths
            means = np.array([s['mean_cn'] for s in segments])
            widths = np.array([s['width'] for s in segments])

            # Normalize by mean copy number
            normalized_cn = means / np.mean(means)

            # Calculate deviations from nearest integer (Eq. 3 in paper)
            deviations = np.abs(normalized_cn - np.round(normalized_cn))

            # Calculate weighted sum of deviations (Eq. 4 in paper)
            total_width = np.sum(widths)
            weighted_deviations = np.sum(deviations * widths) / total_width

            cnh_values[cell] = weighted_deviations

        return cnh_values

    def calculate_cna_burden(self, extended_cnv_file, gene_order_file):
        """
        Calculate CNA burden for each cell in the extended CNV matrix

        Args:
            extended_cnv_file: Path to the extended CNV matrix file
            gene_order_file: Path to the gene order file with genomic positions

        Returns:
            Dictionary with CNA burden results per cell
        """
        # Load gene order file if not already loaded
        if self.gene_lengths is None:
            self.gene_order = self.load_gene_order(gene_order_file)

        # Load extended CNV matrix
        cnv_df = pd.read_csv(extended_cnv_file, sep='\t', index_col=0)

        # First calculate segmented copy numbers
        segmented_data = self.calculate_segmented_copy_numbers(cnv_df)

        # Then calculate CNH for each cell
        cnh_values = self.calculate_cnh(segmented_data)

        # Calculate per-cell burden
        per_cell_burden = {}

        # Process each cell (column in the CNV matrix)
        for cell_id in cnv_df.columns:
            # Get altered genes (where CNV differs from normal 1.0)
            altered_genes = cnv_df.index[cnv_df[cell_id] != 1]

            # Calculate total length of altered regions
            altered_length = sum(self.gene_lengths.get(gene, 0) for gene in altered_genes)

            # Calculate burden percentage
            burden_percent = (altered_length / self.total_genome_length) * 100

            # Store both burden and CNH
            per_cell_burden[cell_id] = {
                'burden': burden_percent,
                'cnh': cnh_values.get(cell_id, np.nan)
            }

        return per_cell_burden

    def calculate_subtype_burdens(self, dataset_id, sample_id, extended_cnv_file, gene_order_file):
        """
        Calculate CNA burden and CNH for each tumor subtype within a sample

        Args:
            dataset_id: Dataset identifier
            sample_id: Sample identifier
            extended_cnv_file: Path to the extended CNV matrix file
            gene_order_file: Path to the gene order file

        Returns:
            Dictionary with CNA burden and CNH results per subtype
        """
        # Load tumor subclusters
        subtypes_df = self.load_tumor_subclusters(dataset_id, sample_id)
        if subtypes_df is None:
            print(f"No subtype information available for {sample_id}, calculating overall burden only")

            # Calculate overall burden without subtype information
            per_cell_burden = self.calculate_cna_burden(extended_cnv_file, gene_order_file)
            burden_values = [x['burden'] for x in per_cell_burden.values()]
            cnh_values = [x['cnh'] for x in per_cell_burden.values()]

            return {
                'overall': {
                    'per_cell_burden': per_cell_burden,
                    'mean_burden': np.mean(burden_values),
                    'std_burden': np.std(burden_values),
                    'mean_cnh': np.mean(cnh_values),
                    'std_cnh': np.std(cnh_values),
                    'total_cells': len(per_cell_burden)
                }
            }

        # Calculate per-cell burden for all cells
        per_cell_burden = self.calculate_cna_burden(extended_cnv_file, gene_order_file)

        # Group cells by subtype
        subtype_burdens = {}

        # Create a mapping from cell barcode to subtype
        barcode_to_subtype = dict(zip(subtypes_df['cell'], subtypes_df['subtype']))

        # Group cells by subtype
        subtype_cells = {}
        for cell, subtype in barcode_to_subtype.items():
            if subtype not in subtype_cells:
                subtype_cells[subtype] = []
            subtype_cells[subtype].append(cell)

        # Calculate burden for each subtype
        for subtype, cells in subtype_cells.items():
            # Filter cells that exist in the CNV matrix
            valid_cells = [cell for cell in cells if cell in per_cell_burden]

            if not valid_cells:
                print(f"Warning: No matching cells found for subtype {subtype}")
                continue

            subtype_burden_values = [per_cell_burden[cell]['burden'] for cell in valid_cells]
            subtype_cnh_values = [per_cell_burden[cell]['cnh'] for cell in valid_cells]

            subtype_burdens[subtype] = {
                'per_cell_burden': {cell: per_cell_burden[cell]['burden'] for cell in valid_cells},
                'per_cell_cnh': {cell: per_cell_burden[cell]['cnh'] for cell in valid_cells},
                'mean_burden': np.mean(subtype_burden_values),
                'std_burden': np.std(subtype_burden_values),
                'mean_cnh': np.mean(subtype_cnh_values),
                'std_cnh': np.std(subtype_cnh_values),
                'total_cells': len(valid_cells)
            }

        # Add overall burden across all cells as a reference
        all_burden_values = [x['burden'] for x in per_cell_burden.values()]
        all_cnh_values = [x['cnh'] for x in per_cell_burden.values()]
        subtype_burdens['overall'] = {
            'per_cell_burden': {cell: x['burden'] for cell, x in per_cell_burden.items()},
            'per_cell_cnh': {cell: x['cnh'] for cell, x in per_cell_burden.items()},
            'mean_burden': np.mean(all_burden_values),
            'std_burden': np.std(all_burden_values),
            'mean_cnh': np.mean(all_cnh_values),
            'std_cnh': np.std(all_cnh_values),
            'total_cells': len(per_cell_burden)
        }

        return subtype_burdens


def write_subtype_summary_csv(results, output_file="cna_burden_subtype_summary.csv"):
    """Write subtype burden and CNH results to a CSV file."""
    summary_data = []

    for sample_id, sample_results in results.items():
        for subtype, result in sample_results.items():
            summary_data.append({
                'Sample_ID': sample_id,
                'Subtype': subtype,
                'Mean_Burden(%)': result['mean_burden'],
                'Std_Burden(%)': result['std_burden'],
                'Mean_CNH': result['mean_cnh'],
                'Std_CNH': result['std_cnh'],
                'Number_of_Cells': result['total_cells']
            })

    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)
    print(f"\nSubtype summary results saved to {output_file}")
    return df


def plot_subtype_metrics(results, output_prefix=None):
    """
    Create plots showing CNA burden and CNH by tumor subtype across samples.
    Each dot represents a subtype's mean value.

    Args:
        results: Dictionary with CNA burden and CNH results per sample and subtype
        output_prefix: Optional prefix for output files
    """
    # Prepare data for plotting
    plot_data = []

    for sample_id, sample_results in results.items():
        # Get a shortened sample ID for display
        short_sample_id = sample_id.split('_')[0]  # Use only the part before the underscore

        for subtype, result in sample_results.items():
            # Skip overall results if there are subtypes
            if subtype == 'overall' and len(sample_results) > 1:
                continue

            plot_data.append({
                'Sample': short_sample_id,
                'Subtype': subtype,
                'Mean_CNA_Burden': result['mean_burden'],
                'Mean_CNH': result['mean_cnh'],
                'Number_of_Cells': result['total_cells']
            })

    plot_df = pd.DataFrame(plot_data)

    # Create the burden plot
    plt.figure(figsize=(18, 10))
    ax = sns.boxplot(data=plot_df, x='Sample', y='Mean_CNA_Burden',
                     color='white', width=0.6, showfliers=False)
    ax = sns.stripplot(data=plot_df, x='Sample', y='Mean_CNA_Burden', hue='Subtype',
                       size=8, palette='viridis', jitter=True, dodge=False, alpha=0.7)
    plt.title('CNA Burden by Sample with Subtype Distribution', fontsize=14)
    ax.set_xlabel('Sample ID', fontsize=12)
    ax.set_ylabel('Mean CNA Burden (%)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title='Subtype')
    sns.despine()
    plt.tight_layout()

    if output_prefix:
        burden_file = f"{output_prefix}_burden.png"
        plt.savefig(burden_file, dpi=300, bbox_inches='tight')
        print(f"Burden plot saved to {burden_file}")
    else:
        plt.show()
    plt.close()

    # Create the CNH plot
    plt.figure(figsize=(18, 10))
    ax = sns.boxplot(data=plot_df, x='Sample', y='Mean_CNH',
                     color='white', width=0.6, showfliers=False)
    ax = sns.stripplot(data=plot_df, x='Sample', y='Mean_CNH', hue='Subtype',
                       size=8, palette='viridis', jitter=True, dodge=False, alpha=0.7)
    plt.title('Copy Number Heterogeneity (CNH) by Sample with Subtype Distribution', fontsize=14)
    ax.set_xlabel('Sample ID', fontsize=12)
    ax.set_ylabel('Mean CNH', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title='Subtype')
    sns.despine()
    plt.tight_layout()

    if output_prefix:
        cnh_file = f"{output_prefix}_cnh.png"
        plt.savefig(cnh_file, dpi=300, bbox_inches='tight')
        print(f"CNH plot saved to {cnh_file}")
    else:
        plt.show()
    plt.close()


def main():
    """Run CNV-regulon threshold analysis pipeline with tumor subtypes."""
    # Configuration
    dataset_id = "ccRCC_GBM"
    base_dir = "/work/project/ladcol_020"  # Default base directory

    sample_ids = [
        "C3L-00004-T1_CPT0001540013",
        "C3L-00026-T1_CPT0001500003",
        "C3L-00088-T1_CPT0000870003",
        # "C3L-00096-T1_CPT0001180011",  
        "C3L-00416-T2_CPT0010100001",
        "C3L-00448-T1_CPT0010160004",
        "C3L-00917-T1_CPT0023690004",
        "C3L-01313-T1_CPT0086820004",
        "C3N-00317-T1_CPT0012280004",
        "C3N-00495-T1_CPT0078510004"
    ]

    # Create analyzer instance
    analyzer = CopyNumberMetrics(base_dir)

    # Load gene order file once
    gene_order_file = os.path.join(base_dir, "scCNV", "inferCNV", "hg38_gencode_v27.txt")
    if not os.path.exists(gene_order_file):
        raise FileNotFoundError(f"Gene order file not found: {gene_order_file}")

    analyzer.load_gene_order(gene_order_file)

    # Process each sample
    results = {}

    for sample_id in sample_ids:
        print(f"\nProcessing sample: {sample_id}")

        # Generate paths
        extended_cnv_file = os.path.join(base_dir, "integration_GRN_CNV", dataset_id, sample_id,
                                         "extended_cnv_matrix.tsv")

        # Check if file exists
        if not os.path.exists(extended_cnv_file):
            print(f"Warning: Extended CNV file not found for sample {sample_id}")
            continue

        # Calculate CNA burden and CNH by tumor subtype
        try:
            subtype_results = analyzer.calculate_subtype_burdens(
                dataset_id, sample_id, extended_cnv_file, gene_order_file
            )
            results[sample_id] = subtype_results

            # Print sample results
            print(f"\nResults for {sample_id}:")
            for subtype, result in subtype_results.items():
                print(f"  Subtype: {subtype}")
                print(f"    Mean CNA burden: {result['mean_burden']:.2f}%")
                print(f"    Std CNA burden: {result['std_burden']:.2f}%")
                print(f"    Mean CNH: {result['mean_cnh']:.4f}")
                print(f"    Std CNH: {result['std_cnh']:.4f}")
                print(f"    Number of cells: {result['total_cells']}")

        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}")

    # Write summary to CSV file
    write_subtype_summary_csv(results)

    # Create visualizations
    plot_subtype_metrics(results, output_prefix="cna_metrics")


if __name__ == "__main__":
    main()