#!/bin/bash
#SBATCH --error=statistics.err
#SBATCH --output=statistics.out
#SBATCH -J statistics
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
