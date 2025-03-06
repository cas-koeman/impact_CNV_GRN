#!/usr/bin/env Rscript

# Required libraries
library(Seurat)
library(Matrix)
library(infercnv)

# Function 1: Prepare data for inferCNV analysis
prepare_infercnv_data <- function(
    data.path, metadata.file, dataset_id_prefix,
    aliquot = aliquot, min.genes = 200,
    min.cells = 3, output_dir) {
  # Set seed for reproducibility
  set.seed(42)

  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }
  file_prefix <- paste0(dataset_id_prefix, aliquot)

  # Read and filter raw data
  message("Reading and filtering 10X data...")
  raw <- Read10X(data.dir = data.path)
  filtered_matrix <- raw[rowSums(raw > 0) >= min.cells, colSums(raw > 0) >= min.genes]
  message("Filtered dimensions: ", nrow(filtered_matrix), " genes x ", ncol(filtered_matrix), " cells")

  # Process metadata - handling gzipped files
  message("Processing metadata...")
  if (endsWith(metadata.file, ".gz")) {
    metadata <- read.table(gzfile(metadata.file), sep = "\t", header = TRUE, stringsAsFactors = FALSE)
  } else {
    metadata <- read.table(metadata.file, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
  }

  # Filter metadata based on Aliquot and Merged_barcode containing dataset prefix
  metadata_filtered <- metadata[
    metadata$Aliquot == aliquot & startsWith(metadata$Merged_barcode, dataset_id_prefix),
  ]
  common_barcodes <- intersect(colnames(filtered_matrix), metadata_filtered$Barcode)
  final_matrix <- filtered_matrix[, common_barcodes]
  final_metadata <- metadata_filtered[match(common_barcodes, metadata_filtered$Barcode), ]

  # Prepare cell annotations
  message("Preparing cell annotations...")
  cell_annotations <- data.frame(
    cell = gsub("-1$", ".1", colnames(final_matrix)),
    cell_type = final_metadata$cell_type.harmonized.cancer
  )

  # Filter out cell types with <= 3 cells
  annotation_counts <- table(cell_annotations$cell_type)
  low_count_annotations <- names(annotation_counts[annotation_counts <= 3])
  if (length(low_count_annotations) > 0) {
    warning("Excluding annotations with <= 3 cells: ", paste(low_count_annotations, collapse = ", "))
    cell_annotations <- cell_annotations[!cell_annotations$cell_type %in% low_count_annotations, ]
  }
  unique_cell_types <- unique(cell_annotations$cell_type)
  message("Cell types in annotations: ", paste(unique_cell_types, collapse = ", "))

  # Export files
  message("Exporting expression matrix and annotations...")
  write.table(final_matrix, file.path(output_dir, paste0(file_prefix, "_expression_matrix.txt")),
              sep = "\t", quote = FALSE, col.names = TRUE, row.names = TRUE)
  write.table(cell_annotations, file.path(output_dir, paste0(file_prefix, "_annotations.txt")),
              sep = "\t", quote = FALSE, col.names = FALSE, row.names = FALSE)

  message("Data preparation complete. Files saved in: ", output_dir)
  return(list(matrix = final_matrix, annotations = cell_annotations, cell_types = unique_cell_types))
}

# Function 2: Run inferCNV analysis
run_infercnv_analysis <- function(
    input_matrix, input_annotations, input_gene_annotation,
    output_dir, ref_groups) {

  # Load input matrix
  message("Loading input matrix...")
  raw_counts_matrix <- as.matrix(read.table(input_matrix, header = TRUE, row.names = 1, sep = "\t"))
  if (!is.numeric(raw_counts_matrix)) {
    stop("Error: Input matrix must contain numeric data")
  }

  # Load annotations
  message("Loading annotations...")
  annotations <- read.table(input_annotations, header = TRUE, sep = "\t")

  # Load gene annotation
  message("Loading gene annotation...")
  gene_order <- read.table(input_gene_annotation, header = TRUE, sep = "\t")

  # Verify reference groups
  anno_groups <- unique(annotations[, 2])
  missing_refs <- setdiff(ref_groups, anno_groups)
  if (length(missing_refs) > 0) {
    warning("Some reference cell types not found in annotations: ", paste(missing_refs, collapse = ", "))
  }

  # Create inferCNV object
  message("Creating inferCNV object...")
  infercnv_obj <- CreateInfercnvObject(
    raw_counts_matrix = raw_counts_matrix,
    annotations_file = input_annotations,
    delim = "\t",
    gene_order_file = input_gene_annotation,
    ref_group_names = ref_groups
  )

  # Run inferCNV analysis
  message("Running inferCNV analysis...")
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

  # Extract tumor subclusters
  message("Extracting tumor subclusters...")
  subclusters_tumor <- infercnv_obj@tumor_subclusters$subclusters$cancer
  cell_names <- colnames(infercnv_obj@expr.data)

  # Initialize clustering_infercnv as NULL
  clustering_infercnv <- NULL

  # Iterate through tumor subclusters and create a data frame
  for (subc in names(subclusters_tumor)) {
    clustering_infercnv <- rbind(
      clustering_infercnv,
      data.frame(
        cell = cell_names[subclusters_tumor[[subc]]],
        infercnv = subc
      )
    )
  }

  # Post-process results
  message("Post-processing results...")
  res_inferCNV <- infercnv_obj@expr.data
  res_inferCNV[res_inferCNV == 0.5] <- 0
  res_inferCNV[res_inferCNV %in% c(1.5, 2, 3)] <- 2

  # Save processed results
  message("Saving processed results...")
  saveRDS(res_inferCNV, file = file.path(output_dir, "processed_infercnv_results.rds"))

  # Save subclustering results
  saveRDS(clustering_infercnv, file = file.path(output_dir, "tumor_subclusters.rds"))

  message("InferCNV analysis complete. Results saved in: ", output_dir)

  # Return the clustering results and processed data
  return(list(
    clustering_results = clustering_infercnv,
    processed_data = res_inferCNV
  ))
}

# Function 3: Main pipeline
run_infercnv_pipeline <- function(
    data.path, metadata.file, dataset_id_prefix,
    aliquot = aliquot, output_dir = ".") {
  # Step 1: Prepare data
  message("Starting data preparation...")
  prepared_data <- prepare_infercnv_data(
    data.path = data.path,
    metadata.file = metadata.file,
    dataset_id_prefix = dataset_id_prefix,
    aliquot = aliquot,
    output_dir = output_dir
  )

  # Define reference groups (non-tumor cell types)
  ref_groups <- prepared_data$cell_types[!grepl("Tumor", prepared_data$cell_types)]
  message("Reference groups: ", paste(ref_groups, collapse = ", "))

  # Define input files for inferCNV
  input_matrix <- file.path(output_dir, paste0(dataset_id_prefix, aliquot, "_expression_matrix.txt"))
  input_annotations <- file.path(output_dir, paste0(dataset_id_prefix, aliquot, "_annotations.txt"))
  input_gene_annotation <- "/work/project/ladcol_020/CNV_calling/inferCNV/hg38_gencode_v27.txt"

  # Step 2: Run inferCNV analysis
  message("Starting inferCNV analysis...")
  run_infercnv_analysis(
    input_matrix = input_matrix,
    input_annotations = input_annotations,
    input_gene_annotation = input_gene_annotation,
    output_dir = output_dir,
    ref_groups = ref_groups
  )

  message("Pipeline completed successfully! Output saved in: ", output_dir)
}

# Main script execution
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 4) {
    stop("Usage: Rscript infercnv_pipeline.R <data.path> <metadata.file> <dataset_id_prefix> <output_dir>")
  }

  data.path <- args[1]
  metadata.file <- args[2]
  dataset_id_prefix <- args[3]
  aliquot <- args[4]
  output_dir <- args[5]

  # Run the pipeline
  run_infercnv_pipeline(
    data.path = data.path,
    metadata.file = metadata.file,
    dataset_id_prefix = dataset_id_prefix,
    aliquot = aliquot,
    output_dir = output_dir
  )
}