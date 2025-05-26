#!/bin/bash
#SBATCH --error=statistics.err
#SBATCH --output=statistics.out
#SBATCH -J statistics
#SBATCH -p slim16
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu 64000

# ==== EDIT THIS SECTION TO ADD/REMOVE SAMPLES ====
SAMPLES=(
    "C3L-00004-T1_CPT0001540013"   
)
# ================================================

# Load environment
source ~/.bashrc
source /work/project/ladcol_020/Miniconda3/etc/profile.d/conda.sh
conda activate pyscenic

# Process each sample
for SAMPLE in "${SAMPLES[@]}"; do
    echo "Processing sample: $SAMPLE"
    python scenic_analysis.py --dataset_id ccRCC_GBM --sample_id "$SAMPLE"
    echo "----------------------------------------"
done

echo "Finished processing all samples"