#!/bin/bash
#SBATCH --error=infercnv_pipeline.err
#SBATCH --output=infercnv_pipeline.out
#SBATCH -J inferCNV
#SBATCH -p slim16
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=64000

# Load required modules (if necessary)
bash ~/.bashrc
source /work/project/ladcol_010/miniconda3/etc/profile.d/conda.sh
conda activate r_env2

# Define input variables
DATA_PATH="/work/project/ladcol_020/datasets/ccRCC_GBM/ccRCC_C3N-00495-T1_CPT0078510004/C3N-00495-T1_CPT0078510004_snRNA_ccRCC/outs/raw_feature_bc_matrix"
METADATA_FILE="/work/project/ladcol_020/datasets/ccRCC_GBM/GSE240822_GBM_ccRCC_RNA_metadata_CPTAC_samples.tsv.gz"
DATASET_PREFIX="ccRCC_"
ALIQUOT="CPT0078510004"
OUTPUT_DIR="/work/project/ladcol_020/CNV_calling/inferCNV/ccRCC_GBM/C3N-00495-T1_CPT0078510004"

# Run the R script
Rscript infercnv_pipeline.R "$DATA_PATH" "$METADATA_FILE" "$DATASET_PREFIX" "$ALIQUOT" "$OUTPUT_DIR"

echo "InferCNV analysis complete! Check the $OUTPUT_DIR directory for results."