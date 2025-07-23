#!/bin/bash
#SBATCH --job-name=preprocess_of2
#SBATCH --account=def-galray
#SBATCH --time=01:00:00              #Adjust time if needed
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

# Load required modules
module load python/3.10

# Activate virtual environment
source ~/jupyter1/bin/activate

# Project directory
cd /project/def-galray/sarai/CrossPoint_OF2

#Run preprocessing script
python preprocess.py