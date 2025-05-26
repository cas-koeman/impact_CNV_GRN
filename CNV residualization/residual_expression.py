import pandas as pd
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import numpy as np
import os
import random

# Set seeds for reproducibility (important for consistent random behavior)
random.seed(42)
np.random.seed(42)

# List of sample IDs to process
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

# Base directory where all sample folders are located
base_dir = '/work/project/ladcol_020/integration_GRN_CNV/ccRCC_GBM'


def load_and_filter_data(expression_path, cnv_path):
    """
    Load expression and CNV matrices, apply quality control (QC) filtering, and align them.
    Returns:
        - expression_df: filtered expression matrix (cells x genes)
        - cnv_df: aligned CNV matrix (same dimensions as expression_df)
    """
    # Load expression matrix (rows: genes, columns: cells)
    ex_df = pd.read_csv(expression_path, sep='\t', index_col=0)

    # Identify mitochondrial genes
    mt_genes = [g for g in ex_df.index if g.startswith('MT-')]  

    # Calculate per-cell QC metrics
    genes_per_cell = (ex_df > 0).sum(axis=0)        # Number of genes expressed in each cell
    counts_per_cell = ex_df.sum(axis=0)             # Total expression counts per cell
    percent_mito = (
        ex_df.loc[mt_genes].sum(axis=0) / counts_per_cell
        if mt_genes else pd.Series(0, index=ex_df.columns)
    )

    # Filter out low-quality cells
    cell_filter = (genes_per_cell >= 200) & (percent_mito < 0.15)
    ex_df = ex_df.loc[:, cell_filter]

    # Filter out lowly expressed genes
    gene_filter = (ex_df > 0).sum(axis=1) >= 3
    ex_df = ex_df.loc[gene_filter, :]

    # Load CNV matrix (rows: cells, columns: genes)
    cnv_df = pd.read_csv(cnv_path, sep='\t', index_col=0)

    # Align both matrices to have the same cells and genes
    shared_cells = ex_df.index.intersection(cnv_df.index)
    shared_genes = ex_df.columns.intersection(cnv_df.columns)
    ex_df = ex_df.loc[shared_cells, shared_genes]
    cnv_df = cnv_df.loc[shared_cells, shared_genes]

    return ex_df, cnv_df


def residualize_expression(expression_df, cnv_df):
    """
    Perform linear regression for each gene to remove the CNV effect from gene expression.
    Returns:
        - residuals: matrix of residual expression values (CNV-corrected)
    """
    residuals = pd.DataFrame(index=expression_df.index, columns=expression_df.columns)

    for gene in expression_df.columns:
        # Skip genes with no variability (cannot model with OLS)
        if expression_df[gene].std() == 0 or cnv_df[gene].std() == 0:
            residuals[gene] = expression_df[gene]
            continue

        # Build linear model: expression ~ CNV + intercept
        X = sm.add_constant(cnv_df[gene])  # Add intercept
        model = OLS(expression_df[gene], X)
        results = model.fit()

        # Store residuals (expression values corrected for CNV effect)
        residuals[gene] = results.resid

    return residuals


def process_sample(sample_id):
    """
    Full pipeline for a single sample:
    - Load and filter data
    - Residualize expression matrix
    - Save residual matrix to file
    """
    print(f"\n=== Processing {sample_id} ===")

    # Construct file paths for input and output
    ex_path = os.path.join(base_dir, sample_id, 'raw_count_matrix.txt')
    cnv_path = os.path.join(base_dir, sample_id, 'extended_cnv_matrix.tsv')
    out_path = os.path.join(base_dir, sample_id, 'residual_matrix.txt')

    try:
        # Load and filter expression & CNV matrices
        expr_df, cnv_df = load_and_filter_data(ex_path, cnv_path)

        # Perform residualization
        residual_df = residualize_expression(expr_df, cnv_df)

        # Save result to disk
        residual_df.to_csv(out_path, sep='\t', float_format='%.5f')
        print(f"Saved residuals to {out_path}")

    except Exception as e:
        # Print error and continue with other samples
        print(f"Failed to process {sample_id}: {e}")


if __name__ == '__main__':
    # Process each sample in the list
    for sample_id in sample_ids:
        process_sample(sample_id)
