#!/usr/bin/env Rscript

#' Single-cell Copy Number Variation Analysis Pipeline using inferCNV
#'
#' This script provides functions for preparing data and running inferCNV analysis
#' on single-cell RNA-seq data to identify copy number alterations.

# Required libraries
library(Seurat)
library(Matrix)
library(inferCNV)

# Hardcoded paths
METADATA_FILE <- "/work/project/ladcol_020/datasets/ccRCC_GBM/GSE240822_GBM_ccRCC_RNA_metadata_CPTAC_samples.tsv.gz"
GENE_ANNOTATION_FILE <- "/work/project/ladcol_020/scCNV/inferCNV/hg38_gencode_v27.txt"

# Function 1: Prepare data for inferCNV analysis ----
#' Prepare expression matrix and annotations for inferCNV
#'
#' @param data.path Path to 10X Genomics data directory
#' @param dataset_id Dataset identifier prefix
#' @param sample_id Sample identifier
#' @param min.genes Minimum genes per cell (default: 200)
#' @param min.cells Minimum cells per gene (default: 3)
#' @param output_dir Output directory
#' @return List containing filtered matrix, annotations, and cell types
prepare_infercnv_data <- function(data.path, dataset_id,
                                 sample_id, min.genes = 200,
                                 min.cells = 3, output_dir) {
  set.seed(42)  # For reproducibility

  # Create output directory
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }
  file_prefix <- paste0(dataset_id, sample_id)

  # Read and filter data
  message("Reading and filtering 10X data...")
  raw <- Read10X(data.dir = data.path)
  filtered_matrix <- raw[rowSums(raw > 0) >= min.cells, colSums(raw > 0) >= min.genes]
  message("Filtered dimensions: ", nrow(filtered_matrix), " genes x ",
          ncol(filtered_matrix), " cells")

  # Process metadata
  message("Processing metadata from: ", METADATA_FILE)
  metadata <- if (endsWith(METADATA_FILE, ".gz")) {
    read.table(gzfile(METADATA_FILE), sep = "\t", header = TRUE, stringsAsFactors = FALSE)
  } else {
    read.table(METADATA_FILE, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
  }

  # Filter metadata and matrix
  metadata_filtered <- metadata[
    grepl(paste0("^", sample_id), metadata$GEO.sample) &
      grepl(paste0("^", dataset_id), metadata$Merged_barcode),
  ]
  common_barcodes <- intersect(colnames(filtered_matrix), metadata_filtered$Barcode)
  final_matrix <- filtered_matrix[, common_barcodes]
  final_metadata <- metadata_filtered[match(common_barcodes, metadata_filtered$Barcode), ]

  # Prepare annotations
  message("Preparing cell annotations...")
  cell_annotations <- data.frame(
    cell = gsub("-1$", ".1", colnames(final_matrix)),
    cell_type = final_metadata$cell_type.harmonized.cancer,
    stringsAsFactors = FALSE
  )

  # Filter rare cell types
  annotation_counts <- table(cell_annotations$cell_type)
  low_count_annotations <- names(annotation_counts[annotation_counts <= 3])
  if (length(low_count_annotations) > 0) {
    warning("Excluding annotations with <= 3 cells: ", paste(low_count_annotations, collapse = ", "))
    cell_annotations <- cell_annotations[!cell_annotations$cell_type %in% low_count_annotations, ]
  }
  unique_cell_types <- unique(cell_annotations$cell_type)
  message("Cell types in annotations: ", paste(unique_cell_types, collapse = ", "))

  # Export files
  message("Exporting expression matrix and annotations to: ", output_dir)
  write.table(final_matrix,
              file.path(output_dir, paste0(file_prefix, "_expression_matrix.txt")),
              sep = "\t", quote = FALSE)
  write.table(cell_annotations,
              file.path(output_dir, paste0(file_prefix, "_annotations.txt")),
              sep = "\t", quote = FALSE, col.names = FALSE, row.names = FALSE)

  return(list(
    matrix = final_matrix,
    annotations = cell_annotations,
    cell_types = unique_cell_types
  ))
}

# Function 2: Run inferCNV analysis ----
#' Run inferCNV analysis on prepared data
#'
#' @param input_matrix Path to expression matrix file
#' @param input_annotations Path to annotations file
#' @param output_dir Output directory
#' @param ref_groups Vector of reference cell types
#' @return List containing clustering results and processed data
run_infercnv_analysis <- function(input_matrix, input_annotations,
                                 output_dir, ref_groups) {
  # Load and validate input
  message("Loading input data...")
  raw_counts_matrix <- as.matrix(read.table(input_matrix, header = TRUE, row.names = 1))
  if (!is.numeric(raw_counts_matrix)) stop("Input matrix must contain numeric data")

  annotations <- read.table(input_annotations, header = FALSE)
  gene_order <- read.table(GENE_ANNOTATION_FILE, header = TRUE)

  # Verify reference groups
  anno_groups <- unique(annotations[, 2])
  missing_refs <- setdiff(ref_groups, anno_groups)
  if (length(missing_refs) > 0) {
    warning("Missing reference cell types: ", paste(missing_refs, collapse = ", "))
  }

  # Create and run inferCNV object
  message("Running inferCNV analysis...")
  infercnv_obj <- CreateInfercnvObject(
    raw_counts_matrix = raw_counts_matrix,
    annotations_file = input_annotations,
    gene_order_file = GENE_ANNOTATION_FILE,
    ref_group_names = ref_groups
  )

  infercnv_obj <- infercnv::run(
    infercnv_obj,
    cutoff = 0.1,
    out_dir = output_dir,
    cluster_by_groups = TRUE,
    denoise = TRUE,
    HMM = TRUE,
    HMM_type = "i6",
    analysis_mode = "subclusters",
    num_threads = 4
  )

  # Process results
  message("Processing results...")
  subclusters_tumor <- infercnv_obj@tumor_subclusters$subclusters$cancer
  cell_names <- colnames(infercnv_obj@expr.data)

  clustering_infercnv <- NULL
  if (!is.null(subclusters_tumor)) {
    for (subc in names(subclusters_tumor)) {
      clustering_infercnv <- rbind(
        clustering_infercnv,
        data.frame(
          cell = cell_names[subclusters_tumor[[subc]]],
          infercnv = subc,
          stringsAsFactors = FALSE
        )
      )
    }
  }

  # Simplify CNV calls
  res_inferCNV <- infercnv_obj@expr.data
  res_inferCNV[res_inferCNV == 0.5] <- 0
  res_inferCNV[res_inferCNV %in% c(1.5, 2, 3)] <- 2

  # Save results
  message("Saving results to: ", output_dir)
  saveRDS(res_inferCNV, file.path(output_dir, "processed_infercnv_results.rds"))
  saveRDS(clustering_infercnv, file.path(output_dir, "tumor_subclusters.rds"))

  return(list(
    clustering_results = clustering_infercnv,
    processed_data = res_inferCNV
  ))
}

# Function 3: Main pipeline ----
#' Run complete inferCNV analysis pipeline
#'
#' @param data.path Path to 10X data directory
#' @param dataset_id Dataset identifier prefix
#' @param sample_id Sample identifier
#' @param output_dir Output directory (default: ".")
#' @return None (writes output files)
run_infercnv_pipeline <- function(data.path, dataset_id,
                                 sample_id, output_dir = ".") {
  # Step 1: Prepare data
  message("Starting inferCNV pipeline")
  prepared_data <- prepare_infercnv_data(
    data.path = data.path,
    dataset_id = dataset_id,
    sample_id = sample_id,
    output_dir = output_dir
  )

  # Define reference groups
  ref_groups <- setdiff(prepared_data$cell_types, "Tumor")
  message("\nReference groups: ", paste(ref_groups, collapse = ", "))

  # Step 2: Run inferCNV
  message("\nStarting inferCNV analysis...")
  run_infercnv_analysis(
    input_matrix = file.path(output_dir, paste0(dataset_id, sample_id, "_expression_matrix.txt")),
    input_annotations = file.path(output_dir, paste0(dataset_id, sample_id, "_annotations.txt")),
    output_dir = output_dir,
    ref_groups = ref_groups
  )

  message("Pipeline completed successfully")
  message("Output directory: ", normalizePath(output_dir))
}

# Command-line execution ----
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 3) {
    stop("Usage: Rscript infercnv_pipeline.R <data.path> <dataset_id> <sample_id> [output_dir]")
  }

  data.path <- args[1]
  dataset_id <- args[2]
  sample_id <- args[3]
  output_dir <- if (length(args) > 3) args[4] else "."

  # Run pipeline
  run_infercnv_pipeline(
    data.path = data.path,
    dataset_id = dataset_id,
    sample_id = sample_id,
    output_dir = output_dir
  )
}