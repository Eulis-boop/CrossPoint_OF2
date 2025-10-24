#!/bin/bash
#SBATCH --job-name=train_full_of2
#SBATCH --account=def-galray
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_full_%j.out
#SBATCH --error=logs/train_full_%j.err

module load python/3.10
module load cuda/12.2
source ~/jupyter1/bin/activate

cd /project/def-galray/sarai/CrossPoint_OF2

nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

unset SLURM_STEP_GPUS
export CUDA_VISIBLE_DEVICES=0

python train.py \
  --data_path /scratch/sarai/ObjectFolder_data/processed \
  --exp_name of2_full_run \
  --epochs 100 \
  --batch_size 32 \
  --print_freq 10 \
  --cuda
