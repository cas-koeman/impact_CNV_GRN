#!/bin/bash

#SBATCH --error=/work/project/ladcol_020/utilities/logs/integrated_pipeline.err
#SBATCH --output=/work/project/ladcol_020/utilities/logs/integrated_pipeline.out
#SBATCH -J CNV_GRN_sc_pipeline
#SBATCH -p slim18
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=64000
#SBATCH --time=3-00:00:00

# Sample and dataset configuration
DATASET_ID="ccRCC_GBM"
SAMPLE_ID="C3L-00004-T1_CPT0001540013"

# Base directories
BASE_DIR="/work/project/ladcol_020"
BASE_DATA_DIR="${BASE_DIR}/datasets/${DATASET_ID}"
RAW_DATA_DIR="${BASE_DATA_DIR}/ccRCC_${SAMPLE_ID}/${SAMPLE_ID}_snRNA_ccRCC/outs/raw_feature_bc_matrix"

# Tool-specific paths
COPYKAT_SCRIPT="${BASE_DIR}/scCNV/copyKAT/copykat_pipeline.R"
INFERCNV_SCRIPT="${BASE_DIR}/scCNV/inferCNV/infercnv_pipeline.R"
PYSCENIC_SCRIPT="${BASE_DIR}/scGRNi/RNA/SCENIC/pySCENIC_pipeline.py"

# Output directories
COPYKAT_OUTPUT="${BASE_DIR}/scCNV/copyKAT/${DATASET_ID}/${SAMPLE_ID}"
INFERCNV_OUTPUT="${BASE_DIR}/scCNV/inferCNV/${DATASET_ID}/${SAMPLE_ID}"
PYSCENIC_OUTPUT="${BASE_DIR}/scGRNi/RNA/SCENIC/${DATASET_ID}/${SAMPLE_ID}"

# pySCENIC parameters
CELL_TYPES=("None" "Tumor" "Non-Tumor")
PRUNE_FLAGS=("None")

# ======================================================================
# Function Definitions
# ======================================================================

get_timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

log_start() {
    local timestamp=$(get_timestamp)
    echo "========================================================================"
    echo "[${timestamp}] STARTING: $1"
    echo "========================================================================"
}

log_end() {
    local timestamp=$(get_timestamp)
    local duration=$(( $(date +%s) - $2 ))
    echo "========================================================================"
    echo "[${timestamp}] COMPLETED: $1 (Duration: ${duration} seconds)"
    echo "========================================================================"
    echo ""
}

run_copykat_analysis() {
    local start_time=$(date +%s)
    local step_name="copyKAT Analysis"

    log_start "${step_name}"

    Rscript -e "
    source('${COPYKAT_SCRIPT}')
    run_copykat_pipeline(
      data_path = '${RAW_DATA_DIR}',
      sample_id = '${SAMPLE_ID}',
      output_dir = '${COPYKAT_OUTPUT}'
    )"

    log_end "${step_name}" "${start_time}"
}

run_infercnv_analysis() {
    local start_time=$(date +%s)
    local step_name="inferCNV Analysis"

    log_start "${step_name}"

    Rscript "${INFERCNV_SCRIPT}" \
        "${RAW_DATA_DIR}" \
        "${SAMPLE_ID}" \
        "${INFERCNV_OUTPUT}"

    log_end "${step_name}" "${start_time}"
}

run_pyscenic_analysis() {
    local start_time=$(date +%s)
    local step_name="pySCENIC Analysis"

    log_start "${step_name}"

    for cell_type in "${CELL_TYPES[@]}"; do
        for prune in "${PRUNE_FLAGS[@]}"; do
            local sub_step_name="pySCENIC (${cell_type}, ${prune})"
            local sub_start_time=$(date +%s)

            log_start "${sub_step_name}"

            python "${PYSCENIC_SCRIPT}" \
                "${BASE_DIR}/scGRNi/RNA/SCENIC/" \
                "${BASE_DATA_DIR}" \
                "${DATASET_ID}/" \
                "${SAMPLE_ID}" \
                --cell_type "${cell_type}" \
                --prune "${prune}" \

            log_end "${sub_step_name}" "${sub_start_time}"
        done
    done

    log_end "${step_name}" "${start_time}"
}

# ======================================================================
# Main Pipeline Execution
# ======================================================================
main() {
    local pipeline_start=$(date +%s)
    local timestamp=$(get_timestamp)

    echo "================================================================================"
    echo "[${timestamp}] STARTING PIPELINE for ${SAMPLE_ID} (Dataset: ${DATASET_ID})"
    echo "================================================================================"
    echo ""

    # Create output directories
    mkdir -p "${COPYKAT_OUTPUT}"
    mkdir -p "${INFERCNV_OUTPUT}"
    mkdir -p "${PYSCENIC_OUTPUT}"

    # Activate Conda environments
    source /work/project/ladcol_010/miniconda3/etc/profile.d/conda.sh
    conda activate r_env2
    run_infercnv_analysis
    conda deactivate

    source /work/project/ladcol_020/Miniconda3/etc/profile.d/conda.sh
    conda activate pyscenic
    run_pyscenic_analysis
    conda deactivate

    source /work/project/ladcol_020/Miniconda3/etc/profile.d/conda.sh
    conda activate copyKAT
    run_copykat_analysis
    conda deactivate

    local timestamp=$(get_timestamp)
    local duration=$(( $(date +%s) - ${pipeline_start} ))

    echo "================================================================================"
    echo "[${timestamp}] PIPELINE COMPLETED SUCCESSFULLY (Total duration: ${duration} seconds)"
    echo "================================================================================"
}

# Execute main function
main