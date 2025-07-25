#!/bin/bash
#SBATCH --job-name=evaluate_of2
#SBATCH --account=def-galray
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_test_%j.out
#SBATCH --error=logs/eval_test_%j.err

module load python/3.10
source ~/jupyter1/bin/activate

cd /project/def-galray/sarai/CrossPoint_OF2

python evaluate.py \
  --data_path /scratch/sarai/ObjectFolder_data/processed \
  --exp_name test_run \
  --batch_size 32 \
  --model_path checkpoints/test_run/models/best_point_model.pth \
  --cuda