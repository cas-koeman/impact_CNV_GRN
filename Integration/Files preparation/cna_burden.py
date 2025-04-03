#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict


def calculate_cna_burden(extended_cnv_file, gene_order_file):
    """
    Calculate CNA burden for each cell in the extended CNV matrix

    Args:
        extended_cnv_file: Path to the extended CNV matrix file
        gene_order_file: Path to the gene order file with genomic positions

    Returns:
        Dictionary with CNA burden results including:
        - per_cell_burden: Dictionary of {cell_id: burden_percentage}
        - mean_burden: Mean burden across all cells
        - std_burden: Standard deviation of burden across cells
    """
    # Load gene order file to get genomic positions and lengths
    gene_order = pd.read_csv(gene_order_file, sep='\t', header=None,
                             names=['gene', 'chrom', 'start', 'end'])
    gene_order['length'] = gene_order['end'] - gene_order['start']

    # Create gene length dictionary
    gene_lengths = dict(zip(gene_order['gene'], gene_order['length']))

    # Load extended CNV matrix
    cnv_df = pd.read_csv(extended_cnv_file, sep='\t', index_col=0)

    # Initialize results
    per_cell_burden = {}
    total_genome_length = sum(gene_lengths.values())

    # Process each cell (column in the CNV matrix)
    for cell_id in cnv_df.columns:
        # Get altered genes (where CNV differs significantly from normal 1.0)
        # Using 0.9 and 1.1 as thresholds to account for potential noise
        altered_genes = cnv_df.index[(cnv_df[cell_id] < 0.9) | (cnv_df[cell_id] > 1.1)]

        # Calculate total length of altered regions
        altered_length = sum(gene_lengths.get(gene, 0) for gene in altered_genes)

        # Calculate burden percentage
        burden_percent = (altered_length / total_genome_length) * 100
        per_cell_burden[cell_id] = burden_percent

    # Calculate summary statistics
    burden_values = list(per_cell_burden.values())
    mean_burden = np.mean(burden_values)
    std_burden = np.std(burden_values)

    return {
        'per_cell_burden': per_cell_burden,
        'mean_burden': mean_burden,
        'std_burden': std_burden,
        'total_cells': len(per_cell_burden)
    }


def main():
    """Run CNV-regulon threshold analysis pipeline."""
    # Configuration
    dataset_id = "ccRCC_GBM"
    base_dir = "/work/project/ladcol_020"  # Default base directory

    sample_ids = [
        "C3L-00004-T1_CPT0001540013",
        "C3L-00026-T1_CPT0001500003",
        "C3L-00088-T1_CPT0000870003",
        # "C3L-00096-T1_CPT0001180011",  # Commented out as in original
        "C3L-00416-T2_CPT0010100001",
        "C3L-00448-T1_CPT0010160004",
        "C3L-00917-T1_CPT0023690004",
        "C3L-01313-T1_CPT0086820004",
        "C3N-00317-T1_CPT0012280004",
        "C3N-00495-T1_CPT0078510004"
    ]

    results = {}

    for sample_id in sample_ids:
        print(f"\nProcessing sample: {sample_id}")

        # Generate paths
        paths = {
            "gene_order_file": os.path.join(base_dir, "scCNV", "inferCNV", "hg38_gencode_v27.txt"),
            "extended_cnv_file": os.path.join(base_dir, "integration_GRN_CNV", dataset_id, sample_id,
                                              "extended_cnv_matrix.tsv")
        }

        # Check if files exist
        if not os.path.exists(paths["extended_cnv_file"]):
            print(f"Warning: Extended CNV file not found for sample {sample_id}")
            continue

        if not os.path.exists(paths["gene_order_file"]):
            raise FileNotFoundError(f"Gene order file not found: {paths['gene_order_file']}")

        # Calculate CNA burden
        try:
            result = calculate_cna_burden(paths["extended_cnv_file"], paths["gene_order_file"])
            results[sample_id] = result

            print(f"Results for {sample_id}:")
            print(f"  Mean CNA burden: {result['mean_burden']:.2f}%")
            print(f"  Std CNA burden: {result['std_burden']:.2f}%")
            print(f"  Number of cells: {result['total_cells']}")

        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}")

    # Print summary
    print("\nFinal Summary:")
    print("Sample ID\tMean Burden (%)\tStd Burden (%)\tNumber of Cells")
    for sample_id, result in results.items():
        print(f"{sample_id}\t{result['mean_burden']:.2f}\t{result['std_burden']:.2f}\t{result['total_cells']}")


if __name__ == "__main__":
    main()