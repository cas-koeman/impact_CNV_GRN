#!/bin/bash
#SBATCH --error=copykat_pipeline.err
#SBATCH --output=copykat_pipeline.out
#SBATCH -J copyKAT
#SBATCH -p slim18
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu 64000

# Load environment
bash ~/.bashrc
source /work/project/ladcol_020/Miniconda3/etc/profile.d/conda.sh
conda activate copyKAT

# Set variables for the sample and dataset IDs
DATASET_PREFIX="ccRCC_"
ALIQUOT="C3L-00004-T1_CPT0001540013"

# Generate input and ouput paths
BASE_DATA_DIR="/work/project/ladcol_020/datasets/ccRCC_GBM"
DATA_PATH="${BASE_DATA_DIR}/${ALIQUOT}_snRNA_ccRCC/outs/raw_feature_bc_matrix"

# Create output directory if it doesn't exist
OUTPUT_DIR="$(pwd)/ccRCC_GBM/${DATASET_PREFIX}${ALIQUOT}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run Analysis
echo "Starting copyKAT analysis for ${SAMPLE_ID}_${DATASET_ID}"
echo "Input data: ${DATA_PATH}"
echo "Output directory: ${OUTPUT_DIR}"

# Run the R script
Rscript -e "
source('copykat_pipeline.R')
run_copykat_pipeline(
  data_path = '$DATA_PATH',
  dataset_id_prefix = '$DATASET_PREFIX',
  aliquot = '$ALIQUOT',
  output_dir = '$OUTPUT_DIR'
)"

echo "CopyKAT analysis complete! Check the $OUTPUT_DIR directory for results."