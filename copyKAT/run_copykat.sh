#!/bin/bash
#SBATCH --error=copykat_pipeline.err
#SBATCH --output=copykat_pipeline.out
#SBATCH -J copyKAT
#SBATCH -p slim18
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu 64000

bash ~/.bashrc
source /work/project/ladcol_020/Miniconda3/etc/profile.d/conda.sh
conda activate copyKAT

DATA_PATH="/work/project/ladcol_020/datasets/ccRCC_GBM/ccRCC_C3L-00096-T1_CPT0001180011/C3L-00096-T1_CPT0001180011_snRNA_ccRCC/outs/raw_feature_bc_matrix"
METADATA_FILE="/work/project/ladcol_020/datasets/ccRCC_GBM/GSE240822_GBM_ccRCC_RNA_metadata_CPTAC_samples.tsv.gz"
DATASET_PREFIX="C3L-00096-T1_"
ALIQUOT="CPT0001180011"
OUTPUT_DIR="$(pwd)/ccRCC_GBM/${DATASET_PREFIX}${ALIQUOT}"

mkdir -p "$OUTPUT_DIR"

Rscript -e "
source('copykat_pipeline.R')
run_copykat_pipeline(
 data_path = '$DATA_PATH',
 metadata_file = '$METADATA_FILE',
 dataset_id_prefix = '$DATASET_PREFIX',
 aliquot = '$ALIQUOT',
 output_dir = '$OUTPUT_DIR'
)"

echo "CopyKAT analysis complete! Check the $OUTPUT_DIR directory for results."