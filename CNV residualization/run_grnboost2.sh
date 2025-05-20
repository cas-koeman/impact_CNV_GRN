#!/bin/bash
#SBATCH --error=grnboost2.err
#SBATCH --output=grnboost2.out
#SBATCH -J grnboost2
#SBATCH -p slim16
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu 64000

# Load environment
source ~/.bashrc
source /work/project/ladcol_020/Miniconda3/etc/profile.d/conda.sh

# Use your Python environment
conda activate pyscenic

# Run the code (variables set within the .py file)
python scenic_analysis.py
