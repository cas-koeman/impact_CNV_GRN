#!/usr/bin/env Rscript

#' Single-cell Copy Number Variation Analysis Pipeline
#'
#' This script provides functions for running CopyKAT analysis on single-cell RNA-seq data,
#' annotating results with metadata, and visualizing copy number alterations.

# Required libraries
library(Seurat)
library(copykat)
library(Matrix)
library(ComplexHeatmap)
library(circlize)
library(RColorBrewer)

# Hardcoded metadata file path
METADATA_FILE <- "/work/project/ladcol_020/datasets/ccRCC_GBM/GSE240822_GBM_ccRCC_RNA_metadata_CPTAC_samples.tsv.gz"

# Function 1: Integrated CopyKAT analysis ----
#' Run integrated CopyKAT analysis with metadata filtering
#'
#' @param data.path Path to 10X Genomics data directory
#' @param sample_id Sample identifier for filtering
#' @param genome Reference genome (default: "hg20")
#' @param n.cores Number of cores for parallel processing (default: 4)
#' @param output_dir Output directory (default: ".")
#' @return CopyKAT result object
integrated_copykat <- function(data.path,
                             sample_id, genome = "hg20",
                             n.cores = 4, output_dir = ".") {
  set.seed(42)  # For reproducibility

  # Read and process data
  message("Reading 10X data...")
  print(data.path)
  raw <- Read10X(data.dir = data.path)
  seurat_obj <- CreateSeuratObject(
    counts = raw,
    project = sample_id,
    min.cells = 3,
    min.features = 200
  )
  message("Initial dimensions: ", dim(seurat_obj)[1], " genes x ", dim(seurat_obj)[2], " cells")

  # Process metadata
  message("Processing metadata from: ", METADATA_FILE)
  metadata <- if (endsWith(METADATA_FILE, ".gz")) {
    read.table(gzfile(METADATA_FILE), sep = "\t", header = TRUE, stringsAsFactors = FALSE)
  } else {
    read.table(METADATA_FILE, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
  }

  # More robust filtering
  matches <- grepl(paste0("^", sample_id), metadata$GEO.sample)
      if (sum(matches) == 0) {
        stop("No matching rows found for sample_id: ", sample_id)
      }
  metadata_filtered <- metadata[matches, ]

  matching_barcodes <- intersect(metadata_filtered$Barcode, colnames(seurat_obj))
  seurat_filtered <- subset(seurat_obj, cells = matching_barcodes)
  seurat_filtered$cell_type.harmonized.cancer <- metadata_filtered$cell_type.harmonized.cancer[
    match(matching_barcodes, metadata_filtered$Barcode)
  ]

  message("Filtered dimensions: ", dim(seurat_filtered)[1], " genes x ",
          dim(seurat_filtered)[2], " cells")

  # Identify normal cells
  normal_cell_barcodes <- colnames(seurat_filtered)[
    seurat_filtered$cell_type.harmonized.cancer != "Tumor"
  ]
  message("Normal cells identified: ", length(normal_cell_barcodes))

  # Run CopyKAT
  message("Running CopyKAT analysis...")
  copykat(
    rawmat = as.matrix(seurat_filtered@assays$RNA@counts),
    id.type = "S",
    ngene.chr = 5,
    win.size = 25,
    KS.cut = 0.1,
    sam.name = file.path(output_dir, sample_id),
    distance = "euclidean",
    norm.cell.names = normal_cell_barcodes,
    output.seg = TRUE,
    plot.genes = TRUE,
    genome = genome,
    n.cores = n.cores
  )
}

# Function 2: Annotate predictions ----
#' Annotate CopyKAT predictions with metadata
#'
#' @param prediction_file Path to CopyKAT prediction file
#' @param output_file Path for output file
#' @param sample_id Sample identifier for filtering
#' @return Annotated data.frame
annotate_predictions_with_metadata <- function(prediction_file, output_file,
                                             sample_id) {
  # Load data
  message("Loading metadata from: ", METADATA_FILE)
  metadata <- if (endsWith(METADATA_FILE, ".gz")) {
    read.table(gzfile(METADATA_FILE), sep = "\t", header = TRUE, stringsAsFactors = FALSE)
  } else {
    read.table(METADATA_FILE, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
  }

  prediction <- read.table(prediction_file, sep = "\t", header = TRUE, stringsAsFactors = FALSE)

  # More robust filtering
  matches <- grepl(paste0("^", sample_id), metadata$GEO.sample)
      if (sum(matches) == 0) {
        stop("No matching rows found for sample_id: ", sample_id)
      }
  metadata_filtered <- metadata[matches, ]

  # Annotate predictions
  annotation_data <- merge(prediction, metadata_filtered,
                          by.x = "cell.names", by.y = "Barcode",
                          all.x = TRUE)

  # Select and rename columns
  result <- annotation_data[, c("cell.names", "copykat.pred",
                               "Sample_type", "cell_type.harmonized.cancer")]
  names(result)[4] <- "Cell_Type"

  write.table(result, output_file, sep = "\t", row.names = FALSE, quote = FALSE)
  return(result)
}

# Function 3: CNV Heatmap ----
#' Create CNV heatmap with annotations
#'
#' @param cna_file Path to CNA results file
#' @param ploidy_file Path to ploidy annotation file
#' @param output_jpeg Path for output JPEG file
#' @return Invisible NULL
create_cnv_heatmap <- function(cna_file, ploidy_file, output_jpeg) {
  # Read data
  cna_data <- read.table(cna_file, header = TRUE, sep = "\t",
                        stringsAsFactors = FALSE, check.names = FALSE)
  ploidy_data <- read.table(ploidy_file, header = TRUE, sep = "\t",
                           stringsAsFactors = FALSE, check.names = FALSE)

  # Prepare data
  cna_data_subset <- cna_data[, c(1:3, 4:ncol(cna_data))]
  selected_cell_names <- colnames(cna_data_subset)[-c(1:3)]

  ploidy_data$cell.names <- gsub("-1", ".1", ploidy_data$cell.names)
  ploidy_data <- ploidy_data[match(selected_cell_names, ploidy_data$cell.names), ]
  ploidy_data$copykat.pred <- factor(ploidy_data$copykat.pred,
                                    levels = c("diploid", "aneuploid"),
                                    labels = c("Diploid", "Aneuploid"))

  # Create heatmap elements
  mat_data <- t(as.matrix(cna_data_subset[, 4:ncol(cna_data_subset)]))
  cna_data_subset$chrom <- factor(cna_data_subset$chrom,
                                 levels = unique(cna_data_subset$chrom),
                                 labels = paste0("chr", unique(cna_data_subset$chrom)))

  # Define colors and annotations
  my_palette <- colorRampPalette(rev(brewer.pal(n = 3, name = "RdBu")))(999)
  col_breaks <- c(seq(-1, -0.4, length = 50),
                 seq(-0.4, -0.2, length = 150),
                 seq(-0.2, 0.2, length = 600),
                 seq(0.2, 0.4, length = 150),
                 seq(0.4, 1, length = 50))
  col_fun <- colorRamp2(col_breaks[c(1, 100, 400, 700, length(col_breaks))],
                       my_palette[c(1, 100, 400, 700, length(my_palette))])

  ha_ploidy <- rowAnnotation(
    `Ploidy` = ploidy_data$copykat.pred,
    col = list(Ploidy = c("Diploid" = "darkblue", "Aneuploid" = "darkred")),
    show_annotation_name = TRUE
  )

  ha_cell_type <- rowAnnotation(
    `Cell Type` = ploidy_data$Cell_Type,
    col = list(`Cell Type` = c(
      "B-cells" = "#4575B4", "DC" = "#74ADD1", "Endothelial" = "#ABD9E9",
      "Fibroblasts" = "#E9A3A3", "Macrophages" = "#D6604D", "Mast" = "#F4A582",
      "NK" = "#A50F15", "Normal" = "#92C5DE", "Normal epithelial cells" = "#2166AC",
      "Plasma" = "#8C96C6", "T-cells" = "#CA0020", "Tumor" = "#67001F"
    )),
    show_annotation_name = TRUE
  )

  # Generate and save heatmap
  jpeg(output_jpeg, width = 12, height = 10, units = "in", res = 300)
  draw(Heatmap(
    mat_data,
    name = "CNV",
    col = col_fun,
    left_annotation = ha_ploidy,
    right_annotation = ha_cell_type,
    cluster_rows = TRUE,
    clustering_distance_rows = "euclidean",
    clustering_method_rows = "ward.D2",
    show_row_names = FALSE,
    cluster_columns = FALSE,
    column_split = cna_data_subset$chrom,
    column_gap = unit(0.5, "mm"),
    column_title_gp = gpar(fontsize = 8),
    column_title_rot = 90,
    border = TRUE,
    heatmap_legend_param = list(
      title = "Copy Number",
      title_gp = gpar(fontsize = 10),
      labels_gp = gpar(fontsize = 8)
    )
  ))
  dev.off()
}

# Main Pipeline Function ----
#' Run complete CopyKAT analysis pipeline
#'
#' @param data_path Path to 10X data directory
#' @param sample_id Sample identifier
#' @param output_dir Output directory (default: ".")
#' @return None (writes output files)
run_copykat_pipeline <- function(data_path,
                               sample_id, output_dir = ".") {
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  # Run analysis
  message("Starting analysis using metadata from: ", METADATA_FILE)
  results <- integrated_copykat(
    data.path = data_path,
    sample_id = sample_id,
    output_dir = output_dir
  )

  # Define output paths
  base_path <- file.path(output_dir, sample_id)
  prediction_file <- paste0(base_path, "_copykat_prediction.txt")
  cna_file <- paste0(base_path, "_copykat_CNA_results.txt")
  annotated_file <- paste0(base_path, "_copykat_prediction_with_metadata.txt")
  heatmap_file <- paste0(base_path, "_copykat_chromosome_heatmap_cell_types.jpg")

  # Annotate and visualize
  annotated_results <- annotate_predictions_with_metadata(
    prediction_file = prediction_file,
    output_file = annotated_file,
    sample_id = sample_id
  )

  create_cnv_heatmap(
    cna_file = cna_file,
    ploidy_file = annotated_file,
    output_jpeg = heatmap_file
  )

  message("Pipeline completed. Outputs saved in: ", output_dir)
  message(" - Predictions: ", prediction_file)
  message(" - CNA results: ", cna_file)
  message(" - Annotated results: ", annotated_file)
  message(" - Heatmap: ", heatmap_file)
}