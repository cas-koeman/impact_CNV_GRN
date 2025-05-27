#!/bin/bash
#SBATCH --error=cnv_analysis.err
#SBATCH --output=cnv_analysis.out
#SBATCH -J statistics
#SBATCH -p slim16
#SBATCH --ntasks=8
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
    python cnv_scenic_analysis.py --dataset_id ccRCC_GBM --sample_id "$SAMPLE"
    echo "----------------------------------------"
done

echo "Finished processing all samples"