#!/usr/bin/env Rscript

# Required libraries
library(Seurat)
library(copykat)
library(Matrix)
library(ComplexHeatmap)
library(circlize)
library(RColorBrewer)

# Function 1: Integrated CopyKAT analysis
integrated_copykat <- function(data.path, metadata.file, dataset_id_prefix,
                             aliquot = "CPT0001540013", genome = "hg20",
                             n.cores = 4, output_dir = ".") {
  # Set seed for reproducibility
  set.seed(42)

  # Read and process raw data
  message("Reading 10X data...")
  raw <- Read10X(data.dir = data.path)
  seurat_obj <- CreateSeuratObject(
    counts = raw,
    project = aliquot,
    min.cells = 3,
    min.features = 200
  )
  message("Initial dimensions: ", dim(seurat_obj)[1], " genes x ", dim(seurat_obj)[2], " cells")

  # Process metadata - handling gzipped files
  message("Processing metadata...")
  if (is.character(metadata.file)) {
    if (endsWith(metadata.file, ".gz")) {
      metadata <- read.table(gzfile(metadata.file), sep = "\t", header = TRUE, stringsAsFactors = FALSE)
    } else {
      metadata <- read.table(metadata.file, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
    }
  } else {
    metadata <- metadata.file
  }

  # Filter metadata based on Aliquot and Merged_barcode containing dataset prefix
  metadata_filtered <- metadata[
    metadata$Aliquot == aliquot &
    grepl(dataset_id_prefix, metadata$Merged_barcode),
  ]

  # Get matching barcodes and filter Seurat object
  matching_barcodes <- metadata_filtered$Barcode[metadata_filtered$Barcode %in% colnames(seurat_obj)]
  seurat_filtered <- subset(seurat_obj, cells = matching_barcodes)

  # Add cell type annotations
  metadata_filtered <- metadata_filtered[match(matching_barcodes, metadata_filtered$Barcode), ]
  seurat_filtered$cell_type.harmonized.cancer <- metadata_filtered$cell_type.harmonized.cancer

  message("After filtering - dimensions: ", dim(seurat_filtered)[1], " genes x ",
          dim(seurat_filtered)[2], " cells")

  # Get normal cell barcodes (cells that are not labeled as 'Tumor')
  normal_cell_barcodes <- colnames(seurat_filtered)[
    seurat_filtered$cell_type.harmonized.cancer != "Tumor"
  ]
  message("Number of normal cells identified: ", length(normal_cell_barcodes))

  # Extract raw count matrix for CopyKat
  exp.rawdata <- as.matrix(seurat_filtered@assays$RNA@counts)

  # Run CopyKat analysis with normal cells specified
  message("Running CopyKat analysis...")
  copykat.result <- copykat(
    rawmat = exp.rawdata,
    id.type = "S",
    ngene.chr = 5,
    win.size = 25,
    KS.cut = 0.1,
    sam.name = file.path(output_dir, aliquot),
    distance = "euclidean",
    norm.cell.names = normal_cell_barcodes,
    output.seg = TRUE,
    plot.genes = TRUE,
    genome = genome,
    n.cores = n.cores
  )

  return(copykat.result)
}

# Function 2: Annotate predictions with metadata
annotate_predictions_with_metadata <- function(metadata_file, prediction_file, output_file,
                                             dataset_id_prefix, aliquot) {
  # Load the metadata file (handling gzipped files)
  if (endsWith(metadata_file, ".gz")) {
    metadata <- read.table(gzfile(metadata_file), sep = "\t", header = TRUE, stringsAsFactors = FALSE)
  } else {
    metadata <- read.table(metadata_file, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
  }

  # Load the prediction file
  prediction <- read.table(prediction_file, sep = "\t", header = TRUE, stringsAsFactors = FALSE)

  # Filter metadata for Aliquot and Merged_barcode containing dataset prefix
  filtered_metadata <- metadata[
    metadata$Aliquot == aliquot &
    grepl(dataset_id_prefix, metadata$Merged_barcode),
  ]

  # Initialize vectors to hold Sample_type and Cell_Type
  sample_types <- vector("character", length = nrow(prediction))
  cell_types <- vector("character", length = nrow(prediction))

  # Loop through each barcode in the prediction file
  for (i in seq_len(nrow(prediction))) {
    barcode <- prediction$cell.names[i]

    # Match barcode with filtered metadata
    matched_row <- filtered_metadata[filtered_metadata$Barcode == barcode, ]

    # Assign Sample_type and Cell_Type if a match is found
    if (nrow(matched_row) > 0) {
      sample_types[i] <- matched_row$Sample_type
      cell_types[i] <- matched_row$cell_type.harmonized.cancer
    } else {
      sample_types[i] <- NA
      cell_types[i] <- NA
    }
  }

  # Add Sample_type and Cell_Type to the prediction file
  prediction$Sample_type <- sample_types
  prediction$Cell_Type <- cell_types

  # Select relevant columns for the output
  output <- prediction[, c("cell.names", "copykat.pred", "Sample_type", "Cell_Type")]

  # Save the output file
  write.table(output, output_file, sep = "\t", row.names = FALSE, quote = FALSE)

  # Return the output data for inspection
  return(output)
}

# Function 3: Create CNV heatmap
create_cnv_heatmap <- function(cna_file, ploidy_file, output_jpeg) {
  # Read CNA and Ploidy data
  cna_data <- read.table(cna_file, header = TRUE, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
  ploidy_data <- read.table(ploidy_file, header = TRUE, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)

  # Subset CNA data and extract cell names
  cna_data_subset <- cna_data[, c(1:3, 4:ncol(cna_data))]
  selected_cell_names <- colnames(cna_data_subset)[-c(1:3)]

  # Standardize cell names in ploidy data to match CNA
  ploidy_data$cell.names <- gsub("-1", ".1", ploidy_data$cell.names)

  # Match ploidy data with selected CNA cell names
  ploidy_data <- ploidy_data[match(selected_cell_names, ploidy_data$cell.names), ]
  ploidy_data$copykat.pred <- factor(ploidy_data$copykat.pred,
                                    levels = c("diploid", "aneuploid"),
                                    labels = c("Diploid", "Aneuploid"))

  # Prepare the matrix for the heatmap
  mat_data <- t(as.matrix(cna_data_subset[, 4:ncol(cna_data_subset)]))

  # Define color palette and breaks for CNV values
  my_palette <- colorRampPalette(rev(brewer.pal(n = 3, name = "RdBu")))(999)
  col_breaks <- c(seq(-1, -0.4, length = 50),
                  seq(-0.4, -0.2, length = 150),
                  seq(-0.2, 0.2, length = 600),
                  seq(0.2, 0.4, length = 150),
                  seq(0.4, 1, length = 50))
  col_fun <- colorRamp2(col_breaks[c(1, 100, 400, 700, length(col_breaks))],
                        my_palette[c(1, 100, 400, 700, length(my_palette))])

  # Modify chromosome labels
  cna_data_subset$chrom <- factor(cna_data_subset$chrom,
                                  levels = unique(cna_data_subset$chrom),
                                  labels = paste0("chr", unique(cna_data_subset$chrom)))

  # Create row annotations for Ploidy and Cell Type
  ha_ploidy <- rowAnnotation(
    `Ploidy` = ploidy_data$copykat.pred,
    col = list(Ploidy = c("Diploid" = "lightblue", "Aneuploid" = "coral")),
    show_annotation_name = TRUE, annotation_name_side = "top"
  )

  ha_cell_type <- rowAnnotation(
    `Cell Type` = ploidy_data$Cell_Type,
    col = list(`Cell Type` = c(
      "B-cells" = "#4575B4", "DC" = "#74ADD1", "Endothelial" = "#ABD9E9",
      "Fibroblasts" = "#E9A3A3", "Macrophages" = "#D6604D", "Mast" = "#F4A582",
      "NK" = "#A50F15", "Normal" = "#92C5DE", "Normal epithelial cells" = "#2166AC",
      "Plasma" = "#8C96C6", "T-cells" = "#CA0020", "Tumor" = "#67001F"
    )),
    show_annotation_name = TRUE, annotation_name_side = "top"
  )

  # Generate the heatmap
  ht <- Heatmap(
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
  )

  # Save the heatmap as a JPEG file
  jpeg(output_jpeg, width = 12, height = 10, units = "in", res = 300)
  draw(ht)
  dev.off()
}

# Example usage of the complete pipeline
run_copykat_pipeline <- function(
  data_path,
  metadata_file,
  dataset_id_prefix,
  aliquot,
  output_dir = "."
) {
  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Step 1: Run integrated CopyKAT analysis
  results <- integrated_copykat(
    data.path = data_path,
    metadata.file = metadata_file,
    dataset_id_prefix = dataset_id_prefix,
    aliquot = aliquot,
    output_dir = output_dir
  )

  # Define output file paths
  prediction_file <- file.path(output_dir, paste0(aliquot, "_copykat_prediction.txt"))
  cna_file <- file.path(output_dir, paste0(aliquot, "_copykat_CNA_results.txt"))
  annotated_prediction_file <- file.path(output_dir,
                                       paste0(aliquot, "_copykat_prediction_with_metadata.txt"))
  heatmap_file <- file.path(output_dir,
                           paste0(aliquot, "_copykat_chromosome_heatmap_cell_types.jpg"))

  # Step 2: Annotate predictions with metadata
  annotated_results <- annotate_predictions_with_metadata(
    metadata_file = metadata_file,
    prediction_file = prediction_file,
    output_file = annotated_prediction_file,
    dataset_id_prefix = dataset_id_prefix,
    aliquot = aliquot
  )

  # Step 3: Create and save CNV heatmap
  create_cnv_heatmap(
    cna_file = cna_file,
    ploidy_file = annotated_prediction_file,
    output_jpeg = heatmap_file
  )

  message("Pipeline completed successfully!")
  message("Output files saved in: ", output_dir)
}