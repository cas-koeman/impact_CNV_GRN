#!/bin/bash
#SBATCH --error=pySCENIC_pipeline.err
#SBATCH --output=pySCENIC_pipeline.out
#SBATCH -J pyscenic
#SBATCH -p slim16
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=64000

# Load environment
source ~/.bashrc
source /work/project/ladcol_020/Miniconda3/etc/profile.d/conda.sh

# Activate the Conda environment for pySCENIC
conda activate pyscenic

# Set paths (directly in the shell script)
BASE_FOLDER="/work/project/ladcol_020/scGRNi/RNA/SCENIC/"
DATASET_ID="ccRCC_GBM/"
SAMPLE_ID="C3L-00004-T1_CPT0001540013/"

# Define the values for CELL_TYPE and PRUNE to loop over
CELL_TYPES=("" "Tumor" "Non-Tumor")  # Add or remove values as needed
PRUNE_FLAGS=("" "true" "false")      # Add or remove values as needed

# Loop over all combinations of CELL_TYPE and PRUNE
for CELL_TYPE in "${CELL_TYPES[@]}"; do
    for PRUNE in "${PRUNE_FLAGS[@]}"; do
        echo "Running pipeline with CELL_TYPE=$CELL_TYPE and PRUNE=$PRUNE"

        # Run the Python script with the current combination of flags
        python pySCENIC_pipeline.py "$BASE_FOLDER" "$DATASET_ID" "$SAMPLE_ID" "$CELL_TYPE" "$PRUNE"

        # Check if the Python script executed successfully
        if [[ $? -eq 0 ]]; then
            echo "Pipeline completed successfully for CELL_TYPE=$CELL_TYPE and PRUNE=$PRUNE."
        else
            echo "Error: Pipeline failed for CELL_TYPE=$CELL_TYPE and PRUNE=$PRUNE."
            exit 1  # Exit the script if any run fails (optional)
        fi
    done
done

echo "All pipeline runs completed."