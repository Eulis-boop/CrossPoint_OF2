#!/bin/bash
#SBATCH --job-name=train_test_of2
#SBATCH --account=def-galray
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_test_%j.out
#SBATCH --error=logs/train_test_%j.err

module load python/3.10
source ~/jupyter1/bin/activate

cd /project/def-galray/sarai/CrossPoint_OF2

python train.py \
  --data_path /scratch/sarai/ObjectFolder_data/processed \
  --exp_name test_run \
  --epochs 5 \
  --batch_size 4 \
  --print_freq 1 \
  --cuda