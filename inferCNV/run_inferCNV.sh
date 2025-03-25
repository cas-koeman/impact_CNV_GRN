#!/bin/bash
#SBATCH --error=infercnv_pipeline.err
#SBATCH --output=infercnv_pipeline.out
#SBATCH -J inferCNV
#SBATCH -p slim16
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=64000

# Environment Setup
source ~/.bashrc
source /work/project/ladcol_010/miniconda3/etc/profile.d/conda.sh
conda activate r_env2

# Set variables for the sample and dataset IDs
DATASET_ID="ccRCC_GBM"
SAMPLE_ID="C3L-00004-T1_CPT0001540013"

# Generate input and ouput paths
RAW_DATA_BASE="/work/project/ladcol_020/datasets/${DATASET_ID}"
RESULTS_BASE="/work/project/ladcol_020/scCNV/inferCNV/${DATASET_ID}"

DATA_PATH="${RAW_DATA_BASE}/ccRCC_${SAMPLE_ID}/${SAMPLE_ID}_snRNA_ccRCC/outs/raw_feature_bc_matrix"
OUTPUT_DIR="${RESULTS_BASE}/${SAMPLE_ID}"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Run Analysis
echo "Starting inferCNV analysis for ${SAMPLE_ID}_${DATASET_ID}"
echo "Input data: ${DATA_PATH}"
echo "Output directory: ${OUTPUT_DIR}"

# Run the R script (note: metadata file is now hardcoded in the R script)
Rscript infercnv_pipeline.R \
    "${DATA_PATH}" \
    "${SAMPLE_ID}" \
    "${OUTPUT_DIR}"

# Completion
echo "InferCNV analysis complete!"
echo "Results saved in: ${OUTPUT_DIR}"