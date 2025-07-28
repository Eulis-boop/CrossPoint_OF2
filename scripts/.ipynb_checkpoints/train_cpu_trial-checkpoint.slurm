#!/bin/bash
#SBATCH --job-name=cpu_test_run
#SBATCH --account=def-galray
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/cpu_test_%j.out
#SBATCH --error=logs/cpu_test_%j.err

module load python/3.10
source ~/jupyter1/bin/activate

cd /project/def-galray/sarai/CrossPoint_OF2

python train.py \
  --data_path /scratch/sarai/ObjectFolder_data/processed \
  --exp_name cpu_test_run \
  --epochs 2 \
  --batch_size 4 \
  --print_freq 1
