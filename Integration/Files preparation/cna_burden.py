#!/usr/bin/env python
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
        altered_genes = cnv_df.index[cnv_df[cell_id] != 1]

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


def write_summary_csv(results, output_file="cna_burden_summary.csv"):
    """Write summary results to a CSV file."""
    summary_data = []

    for sample_id, result in results.items():
        summary_data.append({
            'Sample_ID': sample_id,
            'Mean_Burden(%)': result['mean_burden'],
            'Std_Burden(%)': result['std_burden'],
            'Number_of_Cells': result['total_cells']
        })

    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)
    print(f"\nSummary results saved to {output_file}")
    return df


def plot_burden_vs_pvalue(cna_results, pvalue_csv_path, threshold=None, output_file=None):
    """
    Create scatter plot of CNA burden vs. p-values for a specific threshold.

    Args:
        cna_results: Dictionary containing CNA burden results from calculate_cna_burden()
        pvalue_csv_path: Path to the CSV file with p-values from threshold analysis
        threshold: Which CNV threshold to use (e.g., '0.1', '0.5', '1.0')
        output_file: Optional path to save the plot (if None, will display)
    """
    # Load the p-value data
    pvalue_df = pd.read_csv(pvalue_csv_path, index_col=0)

    # Verify the threshold exists
    threshold_col = str(float(threshold))  # Normalize threshold format
    if threshold_col not in pvalue_df.index:
        raise ValueError(
            f"Threshold {threshold} not found in p-value data. Available thresholds: {pvalue_df.index.tolist()}")

    # Prepare the data
    plot_data = []
    for sample_id, burden_data in cna_results.items():
        if sample_id in pvalue_df.columns:
            plot_data.append({
                'Sample': sample_id,
                'Mean_Burden': burden_data['mean_burden'],
                'P_Value': pvalue_df.loc[threshold_col, sample_id]
            })

    plot_df = pd.DataFrame(plot_data)

    # Create the plot
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=plot_df, x='Mean_Burden', y='P_Value', s=100)

    # Add labels and title
    ax.set_xlabel('Mean CNA Burden (%)', fontsize=12)
    ax.set_ylabel(f'p-value (Threshold: {float(threshold) * 100}%)', fontsize=12)
    ax.set_title(
        f'CNA Burden vs. CNV-Regulon Association p-value\n(Threshold: {float(threshold) * 100}% of cells with CNV)',
        fontsize=14)

    # Add significance line and improve styling
    plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.7)
    plt.text(x=ax.get_xlim()[1] * 0.95, y=0.055, s='p=0.05', color='r', ha='right')

    # Add sample labels if there's space
    if len(plot_df) <= 20:  # Only label if not too many points
        for line in range(0, plot_df.shape[0]):
            ax.text(plot_df.Mean_Burden[line] + 0.5, plot_df.P_Value[line],
                    plot_df.Sample[line], horizontalalignment='left',
                    size='small', color='black')

    # Adjust layout
    sns.despine()
    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


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

    # Print summary to console
    print("\nFinal Summary:")
    print("Sample ID\tMean Burden (%)\tStd Burden (%)\tNumber of Cells")
    for sample_id, result in results.items():
        print(f"{sample_id}\t{result['mean_burden']:.2f}\t{result['std_burden']:.2f}\t{result['total_cells']}")

    # Write summary to CSV file
    # write_summary_csv(results)

    # Create scatter plot if p-value data exists
    pvalue_csv = "cnv_threshold_summary.csv"
    if os.path.exists(pvalue_csv):
        print("\nCreating scatter plot of CNA burden vs. p-values...")
        try:
            # Example usage - plot for 50% threshold
            plot_burden_vs_pvalue(results, pvalue_csv, threshold=0.5,
                                  output_file="burden_vs_pvalue_50percent.png")
        except Exception as e:
            print(f"Error creating scatter plot: {str(e)}")
    else:
        print(f"\nWarning: P-value data file not found at {pvalue_csv}. Skipping scatter plot.")


if __name__ == "__main__":
    main()
