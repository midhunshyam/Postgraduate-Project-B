#!/bin/bash
#SBATCH --job-name=B
#SBATCH --output=outputB_%j.txt
#SBATCH --error=errorB_%j.txt
#SBATCH --partition=ampere40
#SBATCH --time=168:00:00

# Activate the conda environment
module load anaconda/conda3
source activate
conda activate /home/22058122/BioClinicalBERT

# Load the PyTorch module with CUDA support
module load PyTorch/Python3.10

# Run the Python file
python3 bert.py
