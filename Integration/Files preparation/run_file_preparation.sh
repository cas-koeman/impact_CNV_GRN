#!/bin/bash
#SBATCH --error=files_prep.err
#SBATCH --output=files_prep.out
#SBATCH -J files_prep
#SBATCH -p slim16
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu 64000

# Load environment
source ~/.bashrc
source /work/project/ladcol_020/Miniconda3/etc/profile.d/conda.sh

# Use your Python environment
conda activate pyscenic

# Set variables
DATASET_ID="ccRCC_GBM"
BASE_DIR="/work/project/ladcol_020"

# List of samples to process (can also pass single sample as argument)
SAMPLES="C3L-00004-T1_CPT0001540013"

# If sample ID provided as argument, use that instead of full list
if [ $# -ge 1 ]; then
  SAMPLES=("$@")
fi

# Process each sample
for SAMPLE_ID in "${SAMPLES[@]}"; do
  echo "Processing sample: $SAMPLE_ID"
  # Replace Rscript with python and use the new Python script
  python files_preparation.py "$DATASET_ID" "$SAMPLE_ID" "$BASE_DIR"

  # Check exit status
  if [ $? -ne 0 ]; then
    echo "Error processing sample $SAMPLE_ID"
    exit 1
  fi
done

echo "All samples processed successfully"