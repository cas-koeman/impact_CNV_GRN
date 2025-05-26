#!/bin/bash
#SBATCH --error=cnv_analysis.err
#SBATCH --output=cnv_analysis.out
#SBATCH -J statistics
#SBATCH -p slim16
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu 64000

# Load environment
source ~/.bashrc
source /work/project/ladcol_020/Miniconda3/etc/profile.d/conda.sh
conda activate pyscenic

# Run the pipeline
python cnv_scenic_analysis.py 