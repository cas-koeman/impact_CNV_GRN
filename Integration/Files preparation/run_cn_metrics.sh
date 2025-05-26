#!/bin/bash
#SBATCH --error=metrics.err
#SBATCH --output=metrics.out
#SBATCH -J cn_metrics
#SBATCH -p slim16
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu 64000

# Load environment
source ~/.bashrc
source /work/project/ladcol_020/Miniconda3/etc/profile.d/conda.sh

# Use your Python environment
conda activate pyscenic

# Run the code (variables set within the .py file)
python cn_metrics.py
