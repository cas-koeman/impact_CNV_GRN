#!/usr/bin/env python
# Pipeline for the preparation of the count matrix and (extended) CNV matrices

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import random
import gzip


def create_count_matrix(
        data_path,
        metadata_file,
        dataset_id_prefix,
        sample_id,
        min_genes=200,
        min_cells=3,
        output_dir=".."
):
    """
    Create count matrix from 10X format data
    """
    random.seed(42)
    os.makedirs(output_dir, exist_ok=True)

    # Read data
    print("Reading data from 10X format...")
    raw = sc.read_10x_mtx(data_path)
    print(f"Raw data dimensions: {raw.shape[1]} genes x {raw.shape[0]} cells")

    # Filter data
    sc.pp.filter_cells(raw, min_genes=min_genes)
    sc.pp.filter_genes(raw, min_cells=min_cells)
    print(f"Filtered matrix dimensions: {raw.shape[1]} genes x {raw.shape[0]} cells")
    print(f"Minimum cells per gene: {min_cells} | Minimum genes per cell: {min_genes}")

    # Read and filter metadata
    print(f"Reading metadata from: {metadata_file}")
    try:
        # Check if file is gzipped
        if metadata_file.endswith('.gz'):
            with gzip.open(metadata_file, 'rt', encoding='utf-8') as f:
                metadata = pd.read_csv(f, sep='\t')
        else:
            metadata = pd.read_csv(metadata_file, sep='\t')
    except Exception as e:
        print(f"Error reading metadata file: {str(e)}")
        raise e

    # Validate metadata columns
    required_columns = ["GEO.sample", "Merged_barcode", "Barcode"]
    for col in required_columns:
        if col not in metadata.columns:
            raise ValueError(f"Column {col} not found in metadata")

    # Filter metadata
    metadata_filtered = metadata[
        (metadata['GEO.sample'].str.startswith(sample_id))
        ]
    print(f"Filtered metadata dimensions: {len(metadata_filtered)} rows")

    # Check if any barcodes found after filtering
    if len(metadata_filtered) == 0:
        print("Warning: No entries found after filtering metadata")
        print(f"Unique sample_id values in metadata: {', '.join(metadata['GEO.sample'].unique())}")
        print(f"Sample of Merged_barcode values: {', '.join(metadata['Merged_barcode'].head().tolist())}")

    # Get common barcodes
    print("Finding common barcodes between filtered matrix and metadata...")
    common_barcodes = set(raw.obs_names).intersection(set(metadata_filtered['Barcode']))
    print(f"Number of common barcodes: {len(common_barcodes)}")

    if len(common_barcodes) == 0:
        print("Warning: No common barcodes found between filtered matrix and metadata")

    # Subset the filtered matrix
    raw = raw[list(common_barcodes), :]
    print(f"Final matrix dimensions: {raw.shape[1]} genes x {raw.shape[0]} cells")

    # Convert to DataFrame for writing
    df = pd.DataFrame(raw.X.toarray(), index=raw.obs_names, columns=raw.var_names)

    # Write raw count matrix to output directory
    output_file = os.path.join(output_dir, "count_matrix.txt")
    print(f"Writing filtered count matrix to: {output_file}")
    df.to_csv(output_file, sep='\t')

    return {"matrix": df, "file_path": output_file}


def create_infercnv_matrix(infercnv_file, output_dir=".."):
    """
    Create inferCNV matrix from output file
    """
    print(f"Parsing inferCNV data from: {infercnv_file}")

    # Check if file exists
    if not os.path.exists(infercnv_file):
        raise FileNotFoundError(f"inferCNV file not found: {infercnv_file}")

    # Read the data from the text file
    print("Reading lines from inferCNV file...")
    with open(infercnv_file, 'r') as file:
        lines = file.readlines()

    if len(lines) <= 1:
        raise ValueError("inferCNV file contains insufficient data (less than 2 lines)")

    # Extract header and clean cell names
    print("Extracting header...")
    header = lines[0].strip().split(' ')
    header = [h.strip('"') for h in header]

    # Process all lines except header
    print("Processing gene data...")
    data = []
    gene_names = []

    for line in lines[1:]:
        parts = line.strip().split()
        gene_name = parts[0].strip('"')
        gene_names.append(gene_name)
        measurements = [float(val) for val in parts[1:]]
        data.append(measurements)

    print(f"Number of genes processed: {len(gene_names)}")

    # Create data frame
    print("Creating data frame...")
    df = pd.DataFrame(data, index=gene_names, columns=header)

    # Reset index to create Gene column
    df = df.reset_index().rename(columns={'index': 'Gene'})

    print(f"Final CNV data frame dimensions: {df.shape[0]} genes x {df.shape[1]} columns")

    # Write CNV matrix to output directory
    output_file = os.path.join(output_dir, "cnv_matrix.tsv")
    print(f"Writing CNV matrix to: {output_file}")
    df.to_csv(output_file, sep='\t', index=False)

    return {"matrix": df, "file_path": output_file}


def process_missing_genes_in_cnv(raw_count_file, cnv_file, gene_order_file, output_dir=".."):
    """
    Process missing genes in CNV data
    """
    output_file = os.path.join(output_dir, "extended_cnv_matrix.tsv")
    print(f"Output will be written to: {output_file}")

    # Load gene order file
    print("Loading gene order file...")
    gene_order_data = pd.read_csv(gene_order_file, sep='\t', header=None)
    gene_order_data.columns = ["gene", "chrom", "start", "end"]

    # Create ordered gene list and gene position mapping
    ordered_genes = gene_order_data['gene'].tolist()
    gene_positions = {}
    for _, row in gene_order_data.iterrows():
        gene_positions[row['gene']] = {
            'chrom': row['chrom'],
            'start': row['start'],
            'end': row['end']
        }

    # Create index mapping for quick lookup
    gene_indices = {gene: i for i, gene in enumerate(ordered_genes)}

    # Load CNV matrix
    print("Loading CNV matrix...")
    with open(cnv_file, 'r') as f:
        cnv_lines = f.readlines()

    cnv_header = cnv_lines[0].strip().split('\t')
    cnv_cells = cnv_header[1:]  # Remove the first empty field

    # Parse CNV data
    cnv_data = {}
    for i in range(1, len(cnv_lines)):
        line_parts = cnv_lines[i].strip().split('\t')
        gene = line_parts[0]
        values = line_parts[1:]
        cnv_data[gene] = values

    # Load raw count matrix genes - TRANSPOSED VERSION
    print("Loading raw count matrix...")
    raw_count_df = pd.read_csv(raw_count_file, sep='\t', index_col=0).T  # Note the .T for transpose

    # Get gene names from index (now correct after transposing)
    raw_count_genes = raw_count_df.index.tolist()
    print(f"Number of genes in raw count matrix: {len(raw_count_genes)}")

    # Get cell barcodes from columns
    raw_count_cells = raw_count_df.columns.tolist()
    print(f"Number of cells in raw count matrix: {len(raw_count_cells)}")

    # Find missing genes
    missing_genes = list(set(raw_count_genes) - set(cnv_data.keys()))
    print(f"Total missing genes: {len(missing_genes)}")

    # Initialize counters for summary
    total_added = 0
    total_left_border = 0
    total_right_border = 0
    total_both_borders = 0

    # Create updated CNV data structure
    updated_cnv_data = cnv_data.copy()
    for missing_gene in missing_genes:
        updated_cnv_data[missing_gene] = ["0"] * len(cnv_cells)

    # Process each cell individually
    for cell_idx in range(len(cnv_cells)):
        cell_id = cnv_cells[cell_idx]
        print(f"\nProcessing cell {cell_id} ({cell_idx + 1}/{len(cnv_cells)})...")

        added_genes = 0
        left_border_genes = 0
        right_border_genes = 0
        both_borders_genes = 0

        # For each missing gene
        for missing_gene in missing_genes:
            if missing_gene not in gene_indices:
                continue

            idx = gene_indices[missing_gene]

            # Find chromosome for missing gene
            chrom = gene_positions[missing_gene]['chrom']

            # Find adjacent genes in the gene order file that exist in cnv_data
            left_gene = None
            right_gene = None

            # Find left gene that exists in cnv_data
            for i in range(idx - 1, -1, -1):
                if i < 0:
                    break
                candidate = ordered_genes[i]
                if (candidate in cnv_data and
                        gene_positions[candidate]['chrom'] == chrom):
                    left_gene = candidate
                    break

            # Find right gene that exists in cnv_data
            for i in range(idx + 1, len(ordered_genes)):
                if i >= len(ordered_genes):
                    break
                candidate = ordered_genes[i]
                if (candidate in cnv_data and
                        gene_positions[candidate]['chrom'] == chrom):
                    right_gene = candidate
                    break

            # Determine if we should include the gene for this cell
            if left_gene is not None and right_gene is not None:
                # Check if CNV values are the same for this cell
                left_value = cnv_data[left_gene][cell_idx]
                right_value = cnv_data[right_gene][cell_idx]

                if left_value == right_value:
                    # Add missing gene with same CNV value as neighbors for this cell
                    updated_cnv_data[missing_gene][cell_idx] = left_value
                    added_genes += 1
                else:
                    # Gene is on a border between two CNV states for this cell
                    both_borders_genes += 1
            elif left_gene is not None:
                # Gene is on the right border for this cell
                right_border_genes += 1
            elif right_gene is not None:
                # Gene is on the left border for this cell
                left_border_genes += 1

        # Update totals
        total_added += added_genes
        total_left_border += left_border_genes
        total_right_border += right_border_genes
        total_both_borders += both_borders_genes

    # Write updated CNV matrix
    print("\nWriting updated CNV matrix...")

    # Open output file for writing
    with open(output_file, 'w') as f:
        # Write header
        header_line = '\t'.join([''] + cnv_cells)
        f.write(header_line + '\n')

        # Only write genes that have at least one non-zero value
        genes_written = 0
        for gene in updated_cnv_data:
            if any(val != "0" for val in updated_cnv_data[gene]):
                gene_line = '\t'.join([gene] + updated_cnv_data[gene])
                f.write(gene_line + '\n')
                genes_written += 1

    # Print final summary
    print("\nFINAL SUMMARY:")
    print(f"Total genes in raw count matrix: {len(set(raw_count_genes))}")
    print(f"Total genes in original CNV matrix: {len(cnv_data)}")
    print(f"Total missing genes: {len(missing_genes)}")
    print(f"Total gene-cell combinations added: {total_added}")
    print(f"Total gene-cell combinations on left borders: {total_left_border}")
    print(f"Total gene-cell combinations on right borders: {total_right_border}")
    print(f"Total gene-cell combinations on borders between CNV states: {total_both_borders}")
    print(f"Total genes written to output file: {genes_written}")

    return output_file


def generate_paths(base_dir, dataset_id, sample_id):
    """
    Generate paths based on parameters
    """
    # Define the base paths
    paths = {
        "data_path": os.path.join(base_dir, "datasets", dataset_id, f"ccRCC_{sample_id}",
                                  f"{sample_id}_snRNA_ccRCC", "outs", "raw_feature_bc_matrix"),
        "metadata_file": os.path.join(base_dir, "datasets", dataset_id,
                                      "GSE240822_GBM_ccRCC_RNA_metadata_CPTAC_samples.tsv.gz"),
        "infercnv_data_file": os.path.join(base_dir, "scCNV", "inferCNV", dataset_id, sample_id,
                                           "infercnv.20_HMM_predHMMi6.leiden.hmm_mode-subclusters.Pnorm_0.5.repr_intensities.observations.txt"),
        "gene_order_file": os.path.join(base_dir, "scCNV", "inferCNV", "hg38_gencode_v27.txt"),
        "output_dir": os.path.join(base_dir, "integration_GRN_CNV", dataset_id, sample_id)
    }

    # Create the output directory if it doesn't exist
    if not os.path.exists(paths["output_dir"]):
        os.makedirs(paths["output_dir"], exist_ok=True)
        print(f"Created output directory: {paths['output_dir']}")

    return paths


def run_cnv_analysis(base_dir, dataset_id, sample_id, min_genes=200, min_cells=3):
    """
    Main execution function
    """
    print(f"Starting CNV Analysis for sample: {sample_id}")

    # Generate paths
    paths = generate_paths(base_dir, dataset_id, sample_id)
    print(f"Output directory: {paths['output_dir']}")

    # Check if input files exist before processing
    for path_name in ["data_path", "metadata_file", "infercnv_data_file", "gene_order_file"]:
        if not os.path.exists(paths[path_name]):
            raise FileNotFoundError(f"Input file does not exist: {paths[path_name]}")

    # Prepare raw count matrix - now writes output directly to the sample's output directory
    print("Preparing filtered count matrix")
    raw_count_result = create_count_matrix(
        data_path=paths["data_path"],
        metadata_file=paths["metadata_file"],
        dataset_id_prefix=dataset_id,
        sample_id=sample_id,
        min_genes=min_genes,
        min_cells=min_cells,
        output_dir=paths["output_dir"]
    )

    print(f"Filtered count matrix written to: {raw_count_result['file_path']}")

    # Parse inferCNV data - now writes output directly to the sample's output directory
    print("Parsing inferCNV data")
    cnv_result = create_infercnv_matrix(
        infercnv_file=paths["infercnv_data_file"],
        output_dir=paths["output_dir"]
    )

    print(f"CNV matrix written to: {cnv_result['file_path']}")

    # Process missing genes and create extended CNV matrix - now writes output directly to sample's output directory
    print("Processing missing genes in CNV data...")
    extended_cnv_file = process_missing_genes_in_cnv(
        raw_count_file=raw_count_result["file_path"],
        cnv_file=cnv_result["file_path"],
        gene_order_file=paths["gene_order_file"],
        output_dir=paths["output_dir"]
    )

    print(f"Extended CNV matrix written to: {extended_cnv_file}")

    print("CNV Analysis completed successfully")

    # Return results for potential further processing
    return {
        "raw_count_matrix": raw_count_result["file_path"],
        "cnv_matrix": cnv_result["file_path"],
        "extended_cnv_matrix": extended_cnv_file,
        "output_dir": paths["output_dir"]
    }

def main():
    """
    Modified main execution function
    """
    # Get command line arguments
    if len(sys.argv) < 3:
        print("Usage: python files_preparation.py <dataset_id> <sample_id> [base_dir]")
        sys.exit(1)

    dataset_id = sys.argv[1]
    sample_id = sys.argv[2]
    base_dir = sys.argv[3] if len(sys.argv) > 3 else "/work/project/ladcol_020"

    print("Starting script execution with parameters:")
    print(f"Dataset ID: {dataset_id}")
    print(f"Sample ID: {sample_id}")
    print(f"Base directory: {base_dir}")

    try:
        print(f"Processing sample: {sample_id}")

        # Run analysis for this sample
        result = run_cnv_analysis(
            base_dir=base_dir,
            dataset_id=dataset_id,
            sample_id=sample_id
        )

        print(f"Successfully processed dataset: {sample_id}")
        return result
    except Exception as e:
        print(f"Error processing dataset {sample_id}: {str(e)}")
        raise e


if __name__ == "__main__":
    main()