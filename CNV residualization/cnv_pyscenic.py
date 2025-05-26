#!/usr/bin/env python

# Standard library imports
import os
import argparse
import subprocess
import glob
import gzip
import random
import pyreadr

# Third-party imports
import numpy as np
import pandas as pd
import scanpy as sc
import loompy as lp
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from statsmodels.api import add_constant
from statsmodels.regression.linear_model import OLS
from scipy import sparse

# Set global configurations
sc.settings.seed = 42  # Set Scanpy's random seed
sc.set_figure_params(dpi=150, fontsize=10, dpi_save=600)
sc.settings.njobs = 20

# Set random seeds for reproducibility
random.seed(42)  # Python random module
np.random.seed(42)  # NumPy module


class DataLoader:
    """Handles loading and initial processing of data."""

    def __init__(self, base_folder, data_folder, dataset_id, sample_id, cell_type=None, pruning=None):
        """
        Initialize the DataLoader.

        Args:
            base_folder (str): Base directory for the analysis.
            data_folder (str): Directory for the dataset folder
            dataset_id (str): Dataset identifier.
            sample_id (str): Sample identifier.
            cell_type (str, optional): Cell type ('Tumor' or 'Non-Tumor').
            pruning (bool, optional): Whether to use pruned or unpruned paths.
        """
        self.paths = self.get_analysis_paths(base_folder, data_folder, dataset_id, sample_id, cell_type, pruning)

    def get_analysis_paths(self, base_folder, data_folder, dataset_id, sample_id, cell_type=None, pruning=None):
        """
        Generate and create necessary file paths for analysis.
        """
        # Allow tumor subcluster names (e.g., Tumor_s1) or standard cell types
        if cell_type and not (cell_type in ['Tumor', 'Non-Tumor'] or cell_type.startswith('Tumor_')):
            raise ValueError("cell_type must be 'Tumor', 'Non-Tumor', a tumor subcluster (Tumor_sX), or None")

        # Define base paths - different for tumor subclusters
        if cell_type and cell_type.startswith('Tumor_'):
            # For tumor subclusters, use the subcluster folder as base output
            base_output = os.path.join("/work/project/ladcol_020/residual_CNV", dataset_id, sample_id, cell_type)
            # AnnData file should still be in the main directory
            anndata_path = os.path.join("/work/project/ladcol_020/residual_CNV", dataset_id, sample_id, 'anndata.h5ad')
        else:
            # Standard case
            base_output = os.path.join("/work/project/ladcol_020/residual_CNV", dataset_id, sample_id)
            anndata_path = os.path.join(base_output, 'anndata.h5ad')

        paths = {
            'base': base_folder,
            'databases': os.path.join(base_folder, "databases"),
            'raw_data': os.path.join(data_folder, dataset_id),
            'output': base_output,
            'figures': os.path.join(base_output, "figures"),
            'tumor_subclusters': os.path.join("/work/project/ladcol_020/scCNV/inferCNV/", dataset_id, sample_id, "tumor_subclusters.rds"),
            'anndata': anndata_path,
            'cnv_matrix': os.path.join("/work/project/ladcol_020/integration_GRN_CNV/", dataset_id, sample_id, "extended_cnv_matrix.tsv"),
        }

        # Ensure all directories exist
        for folder in [paths['output'], paths['figures']]:
            os.makedirs(folder, exist_ok=True)

        # Adjust output path for pruning if applicable
        if pruning is not None:
            pruning_suffix = "pruned" if pruning else "unpruned"
            paths['output'] = os.path.join(paths['output'], pruning_suffix)
            os.makedirs(paths['output'], exist_ok=True)

        suffix = f"{cell_type}_" if cell_type else ""

        # Define specific file paths
        paths.update({
            'human_tfs': os.path.join(paths['databases'], 'allTFs_hg38.txt'),
            'ranking_dbs': ' '.join(glob.glob(os.path.join(paths['databases'], '*.feather'))),
            'motif_annotations': os.path.join(paths['databases'], 'motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl'),
            'raw_matrix': os.path.join(paths['raw_data'],
                                       f'ccRCC_{sample_id}',
                                       f'{sample_id}_snRNA_ccRCC',
                                       'outs',
                                       'raw_feature_bc_matrix'),
            'metadata': os.path.join(paths['raw_data'], 'GSE240822_GBM_ccRCC_RNA_metadata_CPTAC_samples.tsv.gz'),
            'filtered_loom': os.path.join(paths['output'], f'{suffix}filtered_scenic.loom'),
            'adjacencies': os.path.join(paths['output'], f'{suffix}adjacencies.csv'),
            'regulons': os.path.join(paths['output'], f'{suffix}reg.csv'),
            'pyscenic_output': os.path.join(paths['output'], f'{suffix}pyscenic_output.loom'),
            'aucell_mtx': os.path.join(paths['output'], f'{suffix}auc.csv')
        })

        return paths

    def read_expression_data(self):
        """
        Reads 10X Genomics expression data and preprocesses it.

        Returns:
            AnnData: Processed expression data.
        """
        print("Reading expression data from %s", self.paths['raw_matrix'])
        adata = sc.read_10x_mtx(self.paths['raw_matrix'], var_names='gene_symbols', cache=False)
        sc.pp.filter_cells(adata, min_genes=0)

        # Compute additional QC metrics
        adata.obs['n_genes'] = (adata.X > 0).sum(axis=1).A1
        adata.obs['percent_mito'] = np.sum(adata[:, adata.var_names.str.startswith('MT-')].X, axis=1).A1 / np.sum(
            adata.X,
            axis=1).A1
        adata.obs['n_counts'] = adata.X.sum(axis=1).A1

        return adata
    
    def read_cnv_data(self):
        """
        Reads CNV data from a specified file and preprocesses it.
        Returns CNV matrix with cells as rows and genes as columns.
        """
        cnv_df = pd.read_csv(self.paths['cnv_matrix'], sep='\t', index_col=0)
        
        # Convert CNV values
        cnv_df = cnv_df 
        
        # Convert barcode format if needed (same as in expression data)
        cnv_df = cnv_df.T  # Transpose to have genes as rows and cells as columns
        cnv_df.index = cnv_df.index.str.replace(r'\.\d+$', '-1', regex=True)

        print(f"Loaded CNV data with shape: {cnv_df.shape}")
        return cnv_df

    def load_tumor_subclusters(self):
        """
        Load tumor subclusters from RDS file and return a DataFrame with consistent barcode format.

        Returns:
            pd.DataFrame: DataFrame with cell barcodes (converted to match AnnData format) and subcluster assignments
        """
        if not os.path.exists(self.paths['tumor_subclusters']):
            print(f"Tumor subclusters file not found at {self.paths['tumor_subclusters']}")
            return None

        result = pyreadr.read_r(self.paths['tumor_subclusters'])
        df = result[None]  # Extract the DataFrame from OrderedDict

        # Convert barcode format from AAACGAACACTTGAGT.1 to AAACGAACACTTGAGT-1
        df['cell'] = df['cell'].str.replace(r'\.\d+$', '-1', regex=True)

        print(f"Loaded tumor subclusters with {len(df)} cells after barcode conversion")
        return df


class DataPreprocessor:
    """Handles filtering, normalization, and preprocessing of data."""

    def __init__(self, adata, cnv_data, paths):
        """
        Initialize the DataPreprocessor.

        Args:
            adata (AnnData): Annotated data object.
            paths (dict): Dictionary containing file paths.
        """
        self.adata = adata
        self.cnv_data = cnv_data
        self.paths = paths

    def filter_data(self):
        """
        Filters the dataset based on specified thresholds and prints statistics.

        Returns:
            AnnData: Filtered AnnData object.
        """
        print("Filtering data")
        # Removes cells with fewer than 200 genes
        sc.pp.filter_cells(self.adata, min_genes=200)
        # Removes genes detected in fewer than 3 cells
        sc.pp.filter_genes(self.adata, min_cells=3)
        # Filters out cells with high mitochondrial content (>15%)
        self.adata = self.adata[self.adata.obs['percent_mito'] < 0.15, :]

        # Print statistics after filtering
        print("Counts per gene (min - max): %s - %s", np.sum(self.adata.X, axis=0).min(),
              np.sum(self.adata.X, axis=0).max())
        print("Cells detected per gene (min - max): %s - %s", np.sum(self.adata.X > 0, axis=0).min(),
              np.sum(self.adata.X > 0, axis=0).max())
        print("Minimum counts per gene: %s", 3 * 0.01 * self.adata.X.shape[0])
        print("Minimum samples required: %s", 0.01 * self.adata.X.shape[0])

        # Save the filtered data
        self.adata.write(self.paths['anndata'])
        print("Filtered data saved to %s", self.paths['anndata'])
        print("Final number of cells in filtered AnnData: %d", self.adata.n_obs)
        return self.adata
    
    def align_cnv_to_adata(self):
        """
        Align CNV data to match the filtered AnnData object.
        Ensures matching cell and gene order.
        """
        # Get shared cells and genes as lists in adata order
        print(self.adata.obs_names)
        print(self.adata.var_names)
        print(self.cnv_data.index)
        print(self.cnv_data.columns)

        shared_cells = [cell for cell in self.adata.obs_names if cell in self.cnv_data.index]
        shared_genes = [gene for gene in self.adata.var_names if gene in self.cnv_data.columns]

        if not shared_cells or not shared_genes:
            raise ValueError("No overlapping cells or genes between CNV data and expression data")
        
        # Filter AnnData and CNV to the same order
        self.adata = self.adata[shared_cells, shared_genes].copy()
        aligned_cnv = self.cnv_data.loc[shared_cells, shared_genes]

        # Double check that order matches (optional sanity check)
        assert list(self.adata.obs_names) == list(aligned_cnv.index)
        assert list(self.adata.var_names) == list(aligned_cnv.columns)
        
        print(f"After alignment - AnnData shape: {self.adata.shape}")
        print(f"After alignment - CNV matrix shape: {aligned_cnv.shape}")
        
        return self.adata, aligned_cnv
    
    def residualize_expression(self, cnv_matrix):
        """
        Residualizes the expression data against the CNV matrix using linear regression.
        """
        # Prepare expression matrix (genes x cells)
        expr = self.adata.X.T.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X.T
        expr_df = pd.DataFrame(expr, index=self.adata.var_names, columns=self.adata.obs_names)

        # Prepare CNV matrix and align
        try:
            cnv_df = cnv_matrix.T.loc[expr_df.index, expr_df.columns]
        except KeyError as e:
            raise KeyError(f"Matrix alignment failed: {e}")

        residuals = pd.DataFrame(index=expr_df.index, columns=expr_df.columns)

        for i, gene in enumerate(expr_df.index):
            y = expr_df.loc[gene]
            x = cnv_df.loc[gene]

            if y.std() == 0:
                residuals.loc[gene] = y
            elif x.std() == 0:
                residuals.loc[gene] = y - y.mean()
            else:
                model = OLS(y, add_constant(x)).fit()
                residuals.loc[gene] = model.resid

        # Final diagnostics
        print("\nResiduals (first 5 genes x 5 cells):")
        print(residuals.iloc[:5, :5])
        print("\nResiduals distribution:")
        print(residuals.stack().describe())

        # Update AnnData (transpose back to cells x genes)
        self.adata.X = residuals.T.values
        return self.adata

    def add_metadata(self, sample_id):
        """
        Adds metadata to the AnnData object based on a gzipped metadata file.

        This method filters the metadata to rows where the `GEO.sample` column starts with
        `self.sample_id` (partial match), matches barcodes with the AnnData object, and merges
        the filtered metadata into the AnnData object. It also logs the number of cells for
        each cell type after merging.

        Args:
            None (uses instance attributes).

        Returns:
            AnnData: The AnnData object with metadata joined to its `.obs` attribute.
        """
        print("Adding metadata from %s", self.paths['metadata'])

        # Read the metadata file
        with gzip.open(self.paths['metadata'], 'rt') as f:
            metadata = pd.read_csv(f, sep='\t')

        print("Total metadata rows loaded: %d", len(metadata))

        # Filter metadata: Check if GEO.sample starts with self.sample_id (partial match)
        metadata_filtered = metadata[
            metadata['GEO.sample'].str.startswith(sample_id)
        ]

        print("Rows filtered for sample ID '%s': %d", sample_id, len(metadata_filtered))

        # Match barcodes
        matching_barcodes = metadata_filtered['Barcode'].isin(self.adata.obs_names)
        metadata_filtered = metadata_filtered[matching_barcodes]

        print("Rows with matching barcodes: %d", len(metadata_filtered))

        # Filter AnnData object to only include matching barcodes
        self.adata = self.adata[self.adata.obs_names.isin(metadata_filtered['Barcode'])].copy()

        print("AnnData object filtered to %d cells", self.adata.n_obs)

        # Merge metadata with AnnData
        metadata_filtered = metadata_filtered.set_index('Barcode')
        self.adata.obs = self.adata.obs.join(metadata_filtered[['cell_type.harmonized.cancer']], how='left')

        # Print cell type counts
        if 'cell_type.harmonized.cancer' in self.adata.obs:
            cell_type_counts = self.adata.obs['cell_type.harmonized.cancer'].value_counts()
            print("Cell type counts after adding metadata:")
            for cell_type, count in cell_type_counts.items():
                print("%s: %d cells", cell_type, count)
        else:
            print("No 'cell_type.harmonized.cancer' column found in metadata.")

        print("Final number of annotated cells in AnnData: %d", self.adata.n_obs)
        return self.adata

    def preprocess_data(self):
        """
        Robust preprocessing with NaN handling and your exact HVG parameters
        """
        print("Starting preprocessing with NaN checks")
        
        # 1. Convert to dense and ensure float32
        if sparse.issparse(self.adata.X):
            self.adata.X = self.adata.X.toarray()
        self.adata.X = self.adata.X.astype(np.float32)
        
        # 2. Store raw data safely
        self.adata.raw = sc.AnnData(
            X=self.adata.X.copy(),
            var=self.adata.var.copy(),
            obs=self.adata.obs.copy()
        )
        
        # 3. Handle negative residuals (set negatives to zero)
        print("Adjusting residual values")
        if np.any(self.adata.X < 0):
            # Replace negative values with zeros (keeping zeros and positives unchanged)
            self.adata.X = np.maximum(self.adata.X, 0)
        
        # 4. Check for NaNs/Infs before HVG
        print("Running data quality checks")
        if np.isnan(self.adata.X).any() or np.isinf(self.adata.X).any():
            print("Found NaN/Inf values - replacing with zeros")
            self.adata.X = np.nan_to_num(self.adata.X)
        
        # 5. Your exact HVG pipeline with protection
        print("Running HVG detection")
        try:
            # Calculate mean/dispersion manually first as sanity check
            means = np.mean(self.adata.X, axis=0)
            dispersions = np.var(self.adata.X, axis=0) / means
            
            # Only proceed if calculations succeeded
            if not np.any(np.isnan(means)) and not np.any(np.isnan(dispersions)):
                sc.pp.normalize_per_cell(self.adata, counts_per_cell_after=1e4)
                sc.pp.log1p(self.adata)
                
                sc.pp.highly_variable_genes(
                    self.adata,
                    min_mean=0.0125,
                    max_mean=3,
                    min_disp=0.5
                )
                
                # Save plot
                sc.pl.highly_variable_genes(self.adata, show=False)
                plt.savefig(f"{self.paths['figures']}/hvg_plot.png")
                plt.close()
                
                self.adata = self.adata[:, self.adata.var['highly_variable']].copy()
                print(f"Selected {self.adata.n_vars} HVGs")
            else:
                print("Mean/dispersion calculation failed - skipping HVG")
        except Exception as e:
            print(f"HVG failed: {str(e)}")
            print("Using all genes as fallback")
        
        # 6. Regression with validation
        if all(col in self.adata.obs for col in ['n_counts', 'percent_mito']):
            print("Preparing for regression")
            self.adata.obs['n_counts'] = pd.to_numeric(self.adata.obs['n_counts'], errors='coerce')
            self.adata.obs['percent_mito'] = pd.to_numeric(self.adata.obs['percent_mito'], errors='coerce')
            
            if not self.adata.obs[['n_counts', 'percent_mito']].isna().any().any():
                try:
                    sc.pp.regress_out(self.adata, ['n_counts', 'percent_mito'], n_jobs=1)
                except Exception as e:
                    print(f"Regression failed: {str(e)}")
        
        # 7. Final scaling and save
        sc.pp.scale(self.adata, max_value=10)
        self.adata.write(self.paths['anndata'])
        
        return self.adata


class ClusterVisualizer:
    """Handles clustering and visualization of data."""

    def __init__(self, adata, paths):
        """
        Initialize the ClusterVisualizer.

        Args:
            adata (AnnData): Annotated data object.
            paths (dict): Dictionary containing file paths.
        """
        self.adata = adata
        self.paths = paths

    def cluster_and_umap(self, resolution=0.4, clustering_method='louvain'):
        """
        Perform clustering using either Louvain or Leiden method and generate UMAP visualizations.

        Args:
            resolution (float, optional): Resolution parameter for clustering. Defaults to 0.4.
            clustering_method (str, optional): Clustering method to use ('louvain' or 'leiden'). Defaults to 'louvain'.

        Returns:
            AnnData: Annotated data object with clustering and UMAP results.
        """
        print("Clustering data using %s method", clustering_method)
        # Compute neighborhood graph
        sc.pp.neighbors(self.adata, n_neighbors=15, n_pcs=30)

        # Perform clustering
        clustering_method = clustering_method.lower()
        if clustering_method == 'louvain':
            sc.tl.louvain(self.adata, resolution=resolution, random_state=42)
        elif clustering_method == 'leiden':
            sc.tl.leiden(self.adata, resolution=resolution, random_state=42)
        print("%s clustering completed.", clustering_method.capitalize())

        print("Starting UMAP calculation...")
        sc.tl.umap(self.adata, random_state=42)
        print("UMAP calculation completed.")

        # Set figure directory and plot UMAP
        sc.settings.figdir = self.paths['figures']
        sc.pl.umap(self.adata, color=clustering_method, title=f'{clustering_method.capitalize()} Clustering',
                   save=f"_{clustering_method}_expr.png")
        sc.pl.umap(self.adata, color='cell_type.harmonized.cancer', title='Cell Type (Harmonized Cancer)',
                   save="_expr_cell_type.png")

        # Save clustered data
        if self.paths.get('anndata'):
            self.adata.write(self.paths['anndata'])
            print("Clustered data saved to %s", self.paths['anndata'])
        return self.adata


class GRNInference:
    """Handles Gene Regulatory Network (GRN) inference, context inference, and AUCell scoring."""

    def __init__(self, paths):
        """
        Initialize the GRNInference.

        Args:
            paths (dict): Dictionary containing file paths.
        """
        self.paths = paths

    def create_loom_file(self, adata):
        """
        Create a Loom file from an AnnData object containing gene expression data.

        Args:
            adata (AnnData): Annotated data object with expression data.

        Returns:
            None
        """
        print("Creating loom file at %s", self.paths['filtered_loom'])
        row_attrs = {"Gene": np.array(adata.var_names)}
        col_attrs = {
            "CellID": np.array(adata.obs_names),
            "nGene": np.array(np.sum(adata.X > 0, axis=1)).flatten(),
            "nUMI": np.array(np.sum(adata.X, axis=1)).flatten(),
            "expr_louvain": np.array(adata.obs['louvain']),
            "cell_type.harmonized.cancer": np.array(adata.obs['cell_type.harmonized.cancer'])
        }
        lp.create(self.paths['filtered_loom'], adata.X.transpose(), row_attrs, col_attrs)
        print("Loom file created at: %s", self.paths['filtered_loom'])

    def run_grn_inference(self):
        """
        Run Gene Regulatory Network (GRN) inference using PySCENIC.

        Returns:
            dict: Dictionary with the command string and output path for GRN inference.
        """
        command = f"pyscenic grn {self.paths['filtered_loom']} {self.paths['human_tfs']} -o {self.paths['adjacencies']} --num_workers 20"
        print("Running GRN inference with command: %s", command)
        return {
            'command': command,
            'output_path': self.paths['adjacencies']
        }

    def run_ctx_inference(self, pruning=None):
        """
        Run context inference using PySCENIC.

        Args:
            pruning (bool, optional): Whether to use pruning during context inference. If False, the --no_pruning flag is added.
                                      Defaults to None (no flag added).

        Returns:
            dict: Dictionary with the command string and output path for context inference.
        """
        base_command = (
            f"pyscenic ctx {self.paths['adjacencies']} {self.paths['ranking_dbs']} "
            f"--annotations_fname {self.paths['motif_annotations']} "
            f"--expression_mtx_fname {self.paths['filtered_loom']} "
            f"--output {self.paths['regulons']} "
            f"--mask_dropouts --num_workers 20"
        )
        if pruning is False:
            base_command += " --no_pruning"

        print("Running context inference with command: %s", base_command)
        return {
            'command': base_command,
            'output_path': self.paths['regulons']
        }

    def run_aucell(self):
        """
        Run AUCell scoring to evaluate gene regulatory network activity.

        Returns:
            dict: Dictionary with the command string and output path for AUCell scoring.
        """
        command = f"pyscenic aucell {self.paths['filtered_loom']} {self.paths['regulons']} --output {self.paths['pyscenic_output']} --num_workers 20"
        print("Running AUCell with command: %s", command)
        return {
            'command': command,
            'output_path': self.paths['pyscenic_output']
        }


class ResultVisualizer:
    """Handles visualization of results from GRN inference and AUCell."""

    def __init__(self, paths):
        """
        Initialize the ResultVisualizer.

        Args:
            paths (dict): Dictionary containing file paths.
        """
        self.paths = paths

    def plot_gene_distribution(self, nGenesDetectedPerCell, cell_type):
        """
        Plot the distribution of genes per cell and generate UMAP plots for visualization.
        """
        print("Plotting gene distribution for cell type: %s", cell_type)
        # Convert input data to a Pandas Series for easier manipulation
        nGenesDetectedPerCell_series = pd.Series(nGenesDetectedPerCell.flatten())

        # Compute percentiles for reference lines
        percentiles = nGenesDetectedPerCell_series.quantile([0.01, 0.05, 0.10, 0.50, 1.0])

        # Create histogram plot
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        sns.histplot(nGenesDetectedPerCell, kde=False, bins='fd', ax=ax, color='skyblue', edgecolor='black')

        # Add percentile reference lines
        for i, x in enumerate(percentiles):
            ax.axvline(x=x, color='red', linestyle='--')
            ax.text(x=x, y=ax.get_ylim()[1] * 0.9, s=f'{int(x)} ({int(percentiles.index[i] * 100)}%)',
                    color='red', rotation=30, size='x-small', rotation_mode='anchor')

        # Set axis labels and title
        ax.set_xlabel('# of genes')
        ax.set_ylabel('# of cells')
        ax.set_title(f'Genes per cell distribution - {cell_type or "all cells"}')
        fig.tight_layout()

        # Save figure
        plt.savefig(os.path.join(self.paths['figures'], f"{cell_type + '_' if cell_type else ''}gene_distribution.png"))
        plt.close()

    def aucell_dimensionality_reduction(self, aucell_mtx, adata, cell_type):
        """
        Perform dimensionality reduction on AUCell matrix using UMAP.
        """
        print("Performing dimensionality reduction on AUCell matrix")
        # Apply UMAP dimensionality reduction
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.4, n_components=2, random_state=42)
        umap_result = reducer.fit_transform(aucell_mtx)

        # Store results in AnnData object
        adata.obsm['X_umap'] = umap_result

        # Generate and save UMAP plots
        sc.settings.figdir = self.paths['figures']
        sc.pl.umap(adata, color='cell_type.harmonized.cancer', title=f'Cell Type ({cell_type})',
                   save='_auc.png')

        return pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])

    def create_regulon_heatmap(self):
        """
        Create a heatmap of top regulons' activity across cell types.
        """
        print("Creating regulon heatmap")
        # Load AUCell matrix from loom file
        with lp.connect(self.paths['pyscenic_output'], mode='r+', validate=False) as lf:
            aucell_mtx = pd.DataFrame(lf.ca.RegulonsAUC, index=lf.ca.CellID)

        # Load AnnData object
        adata = sc.read_h5ad(self.paths['anndata'])
        common_cells = adata.obs_names.intersection(aucell_mtx.index)

        # Assign cell types and compute mean AUC by cell type
        aucell_mtx['cell_type'] = adata[common_cells].obs['cell_type.harmonized.cancer']
        mean_auc_by_cell_type = aucell_mtx.groupby('cell_type').mean()

        # Normalize and clean data
        mean_auc_by_cell_type = mean_auc_by_cell_type.replace([float('inf'), float('-inf')], pd.NA).dropna(how='all')

        # Z-score normalization
        normalized_scores = (mean_auc_by_cell_type - mean_auc_by_cell_type.mean()) / mean_auc_by_cell_type.std()
        max_abs_value = np.max(np.abs(normalized_scores.values))

        # Create heatmap with clustering
        g = sns.clustermap(normalized_scores, figsize=[12, 6.5], cmap=plt.cm.RdBu_r, xticklabels=False,
                           yticklabels=True,
                           col_cluster=True, row_cluster=True, tree_kws={'linewidths': 0},
                           cbar_kws={'location': 'right', 'label': 'Z-Score Normalized Regulon Activity'},
                           dendrogram_ratio=0.1, cbar_pos=(0.92, .3, .015, 0.4), vmin=-max_abs_value,
                           vmax=max_abs_value)

        # Customize heatmap appearance
        g.ax_heatmap.yaxis.tick_left()
        g.ax_heatmap.grid(False)
        g.ax_heatmap.tick_params(axis='both', which='both', length=0)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

        # Save figure
        plt.savefig(os.path.join(self.paths['figures'], 'regulon_heatmap.png'), bbox_inches="tight")
        plt.close()

    def plot_top_regulons(self, num_top_regulons=30):
        """
        Plot heatmap of top regulons' activity across cell types.
        """
        print("Plotting top %d regulons", num_top_regulons)
        # Load AUCell matrix
        with lp.connect(self.paths['pyscenic_output'], mode='r+', validate=False) as lf:
            aucell_mtx = pd.DataFrame(lf.ca.RegulonsAUC, index=lf.ca.CellID)

        # Select top regulons based on mean expression
        top_regulons = aucell_mtx.mean(axis=0).nlargest(num_top_regulons)
        selected_aucell_mtx = aucell_mtx[top_regulons.index]

        # Z-score normalization
        scaled_aucell_mtx = (selected_aucell_mtx - selected_aucell_mtx.mean()) / selected_aucell_mtx.std()
        scaled_aucell_mtx = scaled_aucell_mtx.replace([float('inf'), float('-inf')], pd.NA).dropna(how='all')
        max_abs_value = np.max(np.abs(scaled_aucell_mtx.values))

        # Create heatmap with clustering
        g = sns.clustermap(scaled_aucell_mtx.T, figsize=[12, 6.5], cmap=plt.cm.RdBu_r, xticklabels=False,
                           yticklabels=True,
                           col_cluster=True, row_cluster=True, tree_kws={'linewidths': 0},
                           cbar_kws={'location': 'right', 'label': 'Z-Score Normalized Regulon Activity'},
                           dendrogram_ratio=0.1, cbar_pos=(0.92, .3, .015, 0.4), vmin=-max_abs_value,
                           vmax=max_abs_value)

        # Customize heatmap appearance
        g.ax_heatmap.yaxis.tick_left()
        g.ax_heatmap.grid(False)
        g.ax_heatmap.tick_params(axis='both', which='both', length=0)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

        # Save figure
        plt.savefig(os.path.join(self.paths['figures'], 'tumor_regulon_heatmap.png'), bbox_inches="tight")
        plt.close()


class WorkflowManager:
    """Manages the overall workflow and integrates all the above classes."""

    def __init__(self, base_folder, data_folder, dataset_id, sample_id, cell_type=None, pruning=None):
        """
        Initialize the WorkflowManager.

        Args:
            base_folder (str): Base directory for the analysis.
            data_folder (str): Directory for the dataset folder
            dataset_id (str): Dataset identifier.
            sample_id (str): Sample identifier.
            cell_type (str, optional): Cell type ('Tumor' or 'Non-Tumor').
            pruning (bool, optional): Whether to use pruned or unpruned paths.
        """
        self.base_folder = base_folder
        self.data_folder = data_folder
        self.dataset_id = dataset_id
        self.sample_id = sample_id
        self.cell_type = cell_type
        self.pruning = pruning

    def preprocessing_workflow(self):
        """
        Main preprocessing workflow with all visualizations.
        """
        print("Starting preprocessing workflow")

        # Initialize DataLoader without cell_type or pruning for preprocessing
        print("Step 1: Loading expression data")
        data_loader = DataLoader(self.base_folder, self.data_folder, self.dataset_id, self.sample_id)

        # Step 1: Load expression data
        adata = data_loader.read_expression_data()
        cnv_data = data_loader.read_cnv_data()

        # Step 2: Filter and preprocess data
        print("Step 2: Filtering and preprocessing data")
        data_preprocessor = DataPreprocessor(adata, cnv_data, data_loader.paths)
        adata = data_preprocessor.filter_data()
        adata, cnv_aligned = data_preprocessor.align_cnv_to_adata()
        adata = data_preprocessor.residualize_expression(cnv_aligned)
        adata = data_preprocessor.add_metadata(self.sample_id)
        adata = data_preprocessor.preprocess_data()

        # Step 3: Cluster data and generate UMAP visualization
        print("Step 3: Clustering data and generating UMAP visualization")
        cluster_visualizer = ClusterVisualizer(adata, data_loader.paths)
        adata = cluster_visualizer.cluster_and_umap(resolution=0.4)

        # Step 4: Save results to the main directory (not subcluster directory)
        main_anndata_path = os.path.join("/work/project/ladcol_020/residual_CNV", self.dataset_id, self.sample_id, 'anndata.h5ad')
        adata.write(main_anndata_path)
        print(f"Preprocessed data saved to {main_anndata_path}")

        print("Preprocessing workflow complete")
        return adata

    def run_pyscenic_workflow(self):
        """
        Run complete PySCENIC workflow with visualizations.
        """
        print("Starting PySCENIC workflow for dataset %s, sample %s, cell type %s", self.dataset_id,
              self.sample_id, self.cell_type)

        # Initialize DataLoader with pruning flag
        data_loader = DataLoader(self.base_folder, self.data_folder, self.dataset_id, self.sample_id, self.cell_type,
                                 self.pruning)

        # Load AnnData object - we know it exists because main() checked
        adata = sc.read_h5ad(data_loader.paths['anndata'])

        # Subset the data based on cell_type
        if self.cell_type:
            subset_conditions = {
                'Tumor': adata.obs['cell_type.harmonized.cancer'] == 'Tumor',
                'Non-Tumor': adata.obs['cell_type.harmonized.cancer'] != 'Tumor'
            }
            if self.cell_type in subset_conditions:
                adata = adata[subset_conditions[self.cell_type]].copy()
                print("Subsetted data for cell type: %s", self.cell_type)
            else:
                print("Invalid cell type specified: %s. No subsetting applied.", self.cell_type)
        else:
            print("No cell type specified. Using the entire dataset.")

        print("Starting PySCENIC workflow for: %s", self.cell_type or 'whole dataset')

        # Step 1: Create the loom file
        print("Step 1: Creating the desired loom file")
        grn_inference = GRNInference(data_loader.paths)
        grn_inference.create_loom_file(adata)

        # Prepare commands
        grn_inference_config = grn_inference.run_grn_inference()
        ctx_inference_config = grn_inference.run_ctx_inference(self.pruning)
        aucell_config = grn_inference.run_aucell()

        # Step 2: Run GRN inference
        print("Step 2: Running GRN inference")
        print("Executing command: %s", grn_inference_config['command'])
        subprocess.run(grn_inference_config['command'], shell=True, check=True)
        adjacencies = pd.read_csv(grn_inference_config['output_path'], index_col=False)

        # Step 3: Run context-specific inference
        print("Step 3: Running context-specific inference")
        print("Executing command: %s", ctx_inference_config['command'])
        subprocess.run(ctx_inference_config['command'], shell=True, check=True)
        regulons = pd.read_csv(ctx_inference_config['output_path'], header=1)

        # Step 4: Visualize gene distribution
        print("Step 4: Visualizing gene distribution")
        gene_counts = np.sum(adata.X > 0, axis=1)
        result_visualizer = ResultVisualizer(data_loader.paths)
        result_visualizer.plot_gene_distribution(gene_counts, self.cell_type)

        # Step 5: Run AUCell
        print("Step 5: Running AUCell")
        print("Executing command: %s", aucell_config['command'])
        subprocess.run(aucell_config['command'], shell=True, check=True)
        lf = lp.connect(aucell_config['output_path'], mode='r+', validate=False)
        aucell_mtx = pd.DataFrame(lf.ca.RegulonsAUC, index=lf.ca.CellID)
        lf.close()

        # Step 6: Dimensionality reduction on AUCell results
        print("Step 6: Dimensionality reduction on AUCell results")
        result_visualizer.aucell_dimensionality_reduction(aucell_mtx, adata, self.cell_type)

        # Step 7: Visualize AUCell results
        print("Step 7: Visualization of the AUCell results")
        if self.cell_type:
            result_visualizer.plot_top_regulons()
        else:
            result_visualizer.create_regulon_heatmap()

        print("Workflow completed successfully.")

    def run_tumor_subcluster_analysis(self):
        """
        Run PySCENIC workflow for each tumor subcluster with output in subcluster folders.
        """
        print("Starting tumor subcluster analysis")

        # Initialize DataLoader to get paths and load tumor subclusters
        data_loader = DataLoader(self.base_folder, self.data_folder, self.dataset_id, self.sample_id)
        tumor_subclusters = data_loader.load_tumor_subclusters()

        if tumor_subclusters is None or len(tumor_subclusters) == 0:
            print("No tumor subclusters found or file not available. Skipping subcluster analysis.")
            return

        # Get unique subclusters
        unique_subclusters = tumor_subclusters['infercnv'].unique()
        print(f"Found {len(unique_subclusters)} tumor subclusters: {unique_subclusters}")

        if len(unique_subclusters) <= 1:
            print("Only one tumor subcluster found. Skipping subcluster analysis.")
            return

        # Load the full AnnData object
        adata = sc.read_h5ad(data_loader.paths['anndata'])

        # Run PySCENIC for each subcluster
        for subcluster in unique_subclusters:
            subcluster_name = f"Tumor_{subcluster.split('_')[-1]}"  # e.g., Tumor_s1

            print(f"\nProcessing subcluster: {subcluster_name}")

            # Get cells for this subcluster (barcodes already converted to match AnnData format)
            subcluster_cells = tumor_subclusters[tumor_subclusters['infercnv'] == subcluster]['cell'].values
            # Find intersection with AnnData barcodes
            valid_cells = set(subcluster_cells).intersection(set(adata.obs_names))

            if len(valid_cells) == 0:
                print(f"No matching cells found in AnnData for subcluster {subcluster_name}. Skipping.")
                continue

            print(f"Running PySCENIC for {len(valid_cells)} cells in {subcluster_name}")

            # Create subset AnnData
            subcluster_adata = adata[adata.obs_names.isin(valid_cells)].copy()

            # Run PySCENIC workflow for this subcluster
            try:
                # The paths will automatically be created in the subcluster folder
                grn_inference = GRNInference(
                    data_loader.get_analysis_paths(
                        self.base_folder,
                        self.data_folder,
                        self.dataset_id,
                        self.sample_id,
                        cell_type=subcluster_name,
                        pruning=self.pruning
                    )
                )

                # Rest of the SCENIC workflow remains the same
                grn_inference.create_loom_file(subcluster_adata)

                # Prepare commands
                grn_inference_config = grn_inference.run_grn_inference()
                ctx_inference_config = grn_inference.run_ctx_inference(self.pruning)
                aucell_config = grn_inference.run_aucell()

                # Run the commands
                subprocess.run(grn_inference_config['command'], shell=True, check=True)
                subprocess.run(ctx_inference_config['command'], shell=True, check=True)
                subprocess.run(aucell_config['command'], shell=True, check=True)

                print(f"Successfully completed PySCENIC for subcluster {subcluster_name}")

            except Exception as e:
                print(f"Error running PySCENIC for subcluster {subcluster_name}: {str(e)}")
                continue


def parse_arguments():
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(description="Run the PySCENIC workflow.")

    # Required arguments
    parser.add_argument("base_folder", type=str, help="Base directory for the analysis.")
    parser.add_argument("data_folder", type=str, help="Base directory for the datasets.")
    parser.add_argument("dataset_id", type=str, help="Identifier for the dataset.")
    parser.add_argument("sample_id", type=str, help="Identifier for the sample.")

    # Optional arguments
    parser.add_argument("--cell_type", type=str, default=None,
                        help="Specific cell type to subset (e.g., 'Tumor' or 'Non-Tumor'). Defaults to all cells.")
    parser.add_argument("--prune", type=str, default=None,
                        help="Whether to use pruning during GRN inference. Options: 'True', 'False', or None (default).")
    parser.add_argument("--run_subclusters", action='store_true',
                        help="Run analysis on tumor subclusters if available.")

    args = parser.parse_args()

    # Handle the prune flag
    if args.prune is not None:
        args.prune = args.prune.lower()  # Convert to lowercase for case-insensitive comparison
        if args.prune == "true":
            args.prune = True
        elif args.prune == "false":
            args.prune = False
        elif args.prune == "none":  # Convert "None" to Python None
            args.prune = None
        else:
            raise ValueError("Invalid value for --prune. Must be 'True', 'False', or 'None'.")

    # Handle the cell_type flag
    if args.cell_type is not None:
        args.cell_type = args.cell_type.lower()  # Convert to lowercase for case-insensitive comparison
        if args.cell_type == "none":  # Convert "None" to Python None
            args.cell_type = None
        elif args.cell_type == "tumor":  # Standardize to "Tumor"
            args.cell_type = "Tumor"
        elif args.cell_type == "non-tumor":  # Standardize to "Non-Tumor"
            args.cell_type = "Non-Tumor"
        else:
            raise ValueError("Invalid value for --cell_type. Must be 'Tumor', 'Non-Tumor', or 'None'.")

    return args


def main():
    """
    Main function to run the workflow.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Log the arguments
    print(f"Base folder: {args.base_folder}")
    print(f"Data folder: {args.data_folder}")
    print(f"Dataset ID: {args.dataset_id}")
    print(f"Sample ID: {args.sample_id}")
    print(f"Cell type: {args.cell_type}")
    print(f"Prune flag: {args.prune}")
    print(f"Run subclusters: {args.run_subclusters}")

    # Initialize workflow manager
    workflow_manager = WorkflowManager(args.base_folder, args.data_folder, args.dataset_id, args.sample_id,
                                       args.cell_type, args.prune)

    # Check if we need to run preprocessing
    anndata_path = os.path.join("/work/project/ladcol_020/residual_CNV", args.dataset_id, args.sample_id, 'anndata.h5ad')
    if not os.path.exists(anndata_path):
        print("AnnData file not found. Running preprocessing workflow...")
        workflow_manager.preprocessing_workflow()
    else:
        print(f"Found existing AnnData file at {anndata_path}")

    # Run PySCENIC workflow
    workflow_manager.run_pyscenic_workflow()

    # Run tumor subcluster analysis
    workflow_manager.run_tumor_subcluster_analysis()

if __name__ == "__main__":
    main()