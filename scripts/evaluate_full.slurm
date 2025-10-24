#!/bin/bash
#SBATCH --job-name=eval_full_of2
#SBATCH --account=def-galray
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_full_%j.out
#SBATCH --error=logs/eval_full_%j.err

module load python/3.10
source ~/jupyter1/bin/activate

cd /project/def-galray/sarai/CrossPoint_OF2

python evaluate.py \
  --exp_name of2_full_run \
  --data_path /scratch/sarai/ObjectFolder_data/processed \
  --point_model_path point_model_epoch99.pth \
  --image_model_path img_model_epoch99.pth \
  --batch_size 64 \
  --cuda \
  --output_dir results/of2_full_run \
  --tsne_plot
