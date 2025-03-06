library(Seurat)

# Prepare data for inferCNV analysis
prepare_infercnv_data <- function(
  data.path,
  metadata.file,
  dataset_id_prefix,
  aliquot,
  min.genes = 200,
  min.cells = 3,
  output_dir = "."
) {
  set.seed(42)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  # Read data
  print("Reading data from 10X format...")
  raw <- Read10X(data.dir = data.path)
  print(paste("Raw data dimensions:", nrow(raw), "genes x", ncol(raw), "cells"))

  filtered_matrix <- raw[
    rowSums(raw > 0) >= min.cells,
    colSums(raw > 0) >= min.genes
  ]
  print(paste("Filtered matrix dimensions:", nrow(filtered_matrix), "genes x", ncol(filtered_matrix), "cells"))
  print(paste("Minimum cells per gene:", min.cells, "| Minimum genes per cell:", min.genes))

  # Read and filter metadata
  print(paste("Reading metadata from:", metadata.file))
  metadata <- tryCatch({
    read.table(
      metadata.file,
      sep = "\t",
      header = TRUE,
      stringsAsFactors = FALSE,
      fileEncoding = "UTF-8"
    )
  }, error = function(e) {
    print(paste("Error reading metadata file:", e$message))
    stop(e)
  })

  if(!"Aliquot" %in% colnames(metadata)) {
    stop("Column 'Aliquot' not found in metadata")
  }

  if(!"Merged_barcode" %in% colnames(metadata)) {
    stop("Column 'Merged_barcode' not found in metadata")
  }

  metadata_filtered <- metadata[
    metadata$Aliquot == aliquot &
    startsWith(metadata$Merged_barcode, dataset_id_prefix),
  ]

  print(paste("Filtered metadata dimensions:", nrow(metadata_filtered), "rows"))

  # Check if any barcodes found after filtering
  if(nrow(metadata_filtered) == 0) {
    print("Warning: No entries found after filtering metadata")
    print(paste("Unique aliquot values in metadata:", paste(unique(metadata$Aliquot), collapse=", ")))
    print(paste("Sample of Merged_barcode values:", paste(head(metadata$Merged_barcode), collapse=", ")))
  }

  # Check barcode column
  print(paste("Checking if 'Barcode' column exists:", "Barcode" %in% colnames(metadata_filtered)))
  if(!"Barcode" %in% colnames(metadata_filtered)) {
    stop("'Barcode' column not found in filtered metadata")
  }

  # Get common barcodes
  print("Finding common barcodes between filtered matrix and metadata...")
  common_barcodes <- intersect(colnames(filtered_matrix), metadata_filtered$Barcode)
  print(paste("Number of common barcodes:", length(common_barcodes)))

  if(length(common_barcodes) == 0) {
    print("Warning: No common barcodes found between filtered matrix and metadata")
  }

  final_matrix <- filtered_matrix[, common_barcodes]
  print(paste("Final matrix dimensions:", nrow(final_matrix), "genes x", ncol(final_matrix), "cells"))

  return(final_matrix)
}

  # Parse CNV data from inferCNV output
  parse_infercnv_data <- function(infercnv_file) {
    print(paste("Parsing inferCNV data from:", infercnv_file))

    # Check if file exists
    if (!file.exists(infercnv_file)) {
      stop(paste("inferCNV file not found:", infercnv_file))
    }

    # Read the data from the text file
    print("Reading lines from inferCNV file...")
    lines <- readLines(infercnv_file)  # This line was missing in your code

    if (length(lines) <= 1) {
      stop("inferCNV file contains insufficient data (less than 2 lines)")
    }

    # Extract header and clean cell names
    print("Extracting header...")
    header <- gsub('^"|"$', '', unlist(strsplit(lines[1], '\" \"')))

    # Process gene names and measurements
    print("Processing gene data...")
    process_line <- function(line) {
      parts <- unlist(strsplit(line, "\\s+"))
      gene_name <- gsub('^"|"$', '', parts[1])
      measurements <- as.numeric(parts[-1])
      list(gene_name = gene_name, measurements = measurements)
    }

    # Process all lines except header
    print("Processing all data lines...")
    processed_data <- lapply(lines[-1], process_line)
    print(paste("Number of genes processed:", length(processed_data)))

    # Extract gene names and measurements
    print("Extracting gene names and creating measurement matrix...")
    gene_names <- sapply(processed_data, `[[`, "gene_name")
    measurement_matrix <- do.call(rbind, lapply(processed_data, `[[`, "measurements"))
    print(paste("Measurement matrix dimensions:", nrow(measurement_matrix), "genes x", ncol(measurement_matrix), "cells"))

    # Create data frame
    print("Creating data frame...")
    df <- data.frame(measurement_matrix, row.names = gene_names)
    colnames(df) <- header

    # Add gene names column
    df <- cbind(Gene = rownames(df), df)
    print(paste("Final CNV data frame dimensions:", nrow(df), "genes x", ncol(df), "columns"))

    return(df)
}

# Main execution function
run_cnv_analysis <- function(
  data_path,
  metadata_file,
  dataset_prefix,
  aliquot,
  infercnv_data_file,
  output_dir
) {
  print("Starting CNV Analysis")
  print(paste("Output directory:", output_dir))

  # Prepare raw count matrix
  print("Preparing raw count matrix")
  raw_count_matrix <- prepare_infercnv_data(
    data.path = data_path,
    metadata.file = metadata_file,
    dataset_id_prefix = dataset_prefix,
    aliquot = aliquot,
    output_dir = output_dir
  )

  # Write raw count matrix
  print("Writing raw count matrix to file...")
  output_file <- file.path(output_dir, "raw_count_matrix.txt")
  write.table(
    raw_count_matrix,
    output_file,
    sep = "\t",
    quote = FALSE,
    col.names = TRUE,
    row.names = TRUE
  )
  print(paste("Raw count matrix written to:", output_file))

  # Parse inferCNV data
  print("Parsing inferCNV data")
  cnv_df <- parse_infercnv_data(infercnv_data_file)

  # Write CNV matrix
  print("Writing CNV matrix to file...")
  output_file <- file.path(output_dir, "cnv_matrix.tsv")
  write.table(
    cnv_df,
    output_file,
    sep = "\t",
    row.names = FALSE,
    quote = FALSE
  )
  print(paste("CNV matrix written to:", output_file))

  print("CNV Analysis completed successfully")

  # Return results for potential further processing
  list(
    raw_count_matrix = raw_count_matrix,
    cnv_matrix = cnv_df
  )
}

#Example usage
print("Starting script execution")
result <- run_cnv_analysis(
  data_path = "/work/project/ladcol_020/datasets/ccRCC_GBM/ccRCC_C3N-00495-T1_CPT0078510004/C3N-00495-T1_CPT0078510004_snRNA_ccRCC/outs/raw_feature_bc_matrix",
  metadata_file = "/work/project/ladcol_020/datasets/ccRCC_GBM/GSE240822_GBM_ccRCC_RNA_metadata_CPTAC_samples.tsv.gz",
  dataset_prefix = "ccRCC_",
  aliquot = "CPT0078510004",
  infercnv_data_file = "/work/project/ladcol_020/CNV_calling/inferCNV/ccRCC_GBM/C3N-00495-T1_CPT0078510004/infercnv.20_HMM_predHMMi6.leiden.hmm_mode-subclusters.Pnorm_0.5.repr_intensities.observations.txt",
  output_dir = "/work/project/ladcol_020/integration_visualization/ccRCC_GBM/C3N-00495-T1_CPT0078510004"
)
print("Script execution completed")