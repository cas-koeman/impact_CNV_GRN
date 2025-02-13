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
INPUT_GENE_ANNOTATION="/work/project/ladcol_020/CNV_calling/inferCNV/hg38_gencode_v27.txt"
INPUT_ANNOTATIONS="/work/project/ladcol_020/CNV_calling/inferCNV/ccRCC_GBM/C3L-00004-T1_CPT0001540013/ccRCC_C3L-00004-T1_CPT0001540013_annotations.txt"
INPUT_MATRIX="/work/project/ladcol_020/CNV_calling/inferCNV/ccRCC_GBM/C3L-00004-T1_CPT0001540013/ccRCC_C3L-00004-T1_CPT0001540013_expression_matrix.txt"
OUTPUT_DIR="/work/project/ladcol_020/CNV_calling/inferCNV/ccRCC_GBM/C3L-00004-T1_CPT0001540013"

# Run the R script
Rscript infercnv_pipeline.R "$INPUT_MATRIX" "$INPUT_ANNOTATIONS" "$INPUT_GENE_ANNOTATION" "$OUTPUT_DIR"

echo "InferCNV analysis complete! Check the $OUTPUT_DIR directory for results."

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
DATA_PATH="/work/project/ladcol_020/CNV_calling/inferCNV/ccRCC_GBM/C3L-00004-T1_CPT0001540013"
METADATA_FILE="/work/project/ladcol_020/CNV_calling/inferCNV/metadata.txt"  # You'll need to specify the correct path
DATASET_PREFIX="ccRCC_"
OUTPUT_DIR="/work/project/ladcol_020/CNV_calling/inferCNV/ccRCC_GBM/C3L-00004-T1_CPT0001540013"

# Run the R script
Rscript infercnv_pipeline.R "$DATA_PATH" "$METADATA_FILE" "$DATASET_PREFIX" "$OUTPUT_DIR"

echo "InferCNV analysis complete! Check the $OUTPUT_DIR directory for results."