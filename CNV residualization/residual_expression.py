import pandas as pd
from statsmodels.regression.linear_model import OLS
from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2
import random
import numpy as np

# Set the seed for reproducibility
random.seed(42)  # Set the Python random seed
np.random.seed(42)  # Set the NumPy random seed

def load_and_filter_data(expression_path, cnv_path):
    """
    1. Load and filter expression matrix (QC)
    2. Align CNV matrix to filtered expression matrix (same cells & genes)
    3. Return both matrices with matching row/column order
    """
    # 1. Load & Filter Expression Matrix 
    print("Loading expression matrix...")
    ex_df = pd.read_csv(expression_path, sep='\t', index_col=0)  # rows=cells, cols=genes
    print(f"Initial expression shape: {ex_df.shape}")

    # QC filtering 
    mt_genes = [g for g in ex_df.index if g.startswith('MT-')]  

    # Calculate QC metrics (all per-cell, so axis=0 for columns)
    genes_per_cell = (ex_df > 0).sum(axis=0)    # Genes detected per cell 
    counts_per_cell = ex_df.sum(axis=0)         # Total counts per cell 
    percent_mito = ex_df.loc[mt_genes].sum(axis=0) / counts_per_cell if mt_genes else pd.Series(0, index=ex_df.columns)

    # Apply cell filters 
    cell_filter = (genes_per_cell >= 200) & (percent_mito < 0.15)
    ex_df = ex_df.loc[:, cell_filter]  # Keep only passing cells 
    print(f"Cells after QC: {ex_df.shape[1]}")  

    # QC filtering (genes) 
    gene_filter = (ex_df > 0).sum(axis=1) >= 3  # Min 3 cells expressing gene
    ex_df = ex_df.loc[gene_filter, :]           # Keep only passing genes
    print(f"Genes after QC: {ex_df.shape[0]}")  

    # 2. Align CNV Matrix to Filtered Expression 
    print("\nLoading and aligning CNV matrix...")
    cnv_df = pd.read_csv(cnv_path, sep='\t', index_col=0)  # rows=cells, cols=genes
    cnv_df = cnv_df * 2 - 2

    # Get shared cells & genes (after expression filtering)
    shared_cells = ex_df.index.intersection(cnv_df.index)
    shared_genes = ex_df.columns.intersection(cnv_df.columns)

    # Subset both matrices to shared dimensions
    ex_df = ex_df.loc[shared_cells, shared_genes]
    cnv_df = cnv_df.loc[shared_cells, shared_genes]

    # 3. Verify Alignment 
    print(f"\nFinal expression shape: {ex_df.shape}")
    print(f"Final CNV shape: {cnv_df.shape}")
    print("\nExpression head:\n", ex_df.iloc[:10, :5])
    print("\nCNV head:\n", cnv_df.iloc[:10, :5])

    return ex_df, cnv_df

def residualize_expression(expression_df, cnv_df):
    """
    Remove CNV effects from expression using linear regression.
    CNV values are centered at 1 using (CNV * 2 - 2), so CNV = 1 becomes 0.
    """
    import statsmodels.api as sm

    residuals = pd.DataFrame(index=expression_df.index, columns=expression_df.columns)
    
    for gene in expression_df.columns:
        # Skip genes with zero variance in either expression or CNV
        if expression_df[gene].std() == 0 or cnv_df[gene].std() == 0:
            residuals[gene] = expression_df[gene]
            continue
        
        # Add constant to model
        X = sm.add_constant(cnv_df[gene])
        model = OLS(expression_df[gene], X)
        results = model.fit()
        
        # Store residuals
        residuals[gene] = results.resid

    print("\nResiduals head:\n", residuals.iloc[:10, :5])
    return residuals


if __name__ == '__main__':
    # Paths
    ex_path = '/work/project/ladcol_020/integration_GRN_CNV/ccRCC_GBM/C3L-01313-T1_CPT0086820004/raw_count_matrix.txt'
    cnv_path = '/work/project/ladcol_020/integration_GRN_CNV/ccRCC_GBM/C3L-01313-T1_CPT0086820004/extended_cnv_matrix.tsv'
    tf_path = '/work/project/ladcol_020/scGRNi/RNA/SCENIC/databases/allTFs_hg38.txt'
    
    # 1. Load and filter data (rows=genes, cols=cells)
    expr_df, cnv_df = load_and_filter_data(ex_path, cnv_path)
    
    # 2. Residualize expression against CNV (rows=genes, cols=cells)
    residual_expr = residualize_expression(expr_df, cnv_df)
    
    # 3. Get TF names
    tf_names = load_tf_names(tf_path)

    # 4. Check overlap between TF names and residual expression genes
    shared_genes = set(residual_expr.index).intersection(tf_names)
    print(f"Found {len(shared_genes)} TFs that are present in both residual expression and TF list.")

    print(residual_expr.index[:5])
    
    # 5. Run GRNBoost2 on residuals 
    # Note: grnboost2 expects cells as rows and genes as columns, so transpose the residual expression DataFrame
    network_res = grnboost2(
        expression_data=residual_expr.T,
        tf_names=tf_names
    )
    
    # Save results
    network_res.to_csv('residualized.tsv', sep='\t', index=False, header=False)

    # 6. Run GRNBoost2 on count matrix 
    # Note: grnboost2 expects cells as rows and genes as columns, so transpose the residual expression DataFrame
    network_counts = grnboost2(
        expression_data=expr_df.T,
        tf_names=tf_names
    )

    # Save results
    network_counts.to_csv('counts.tsv', sep='\t', index=False, header=False)

    
