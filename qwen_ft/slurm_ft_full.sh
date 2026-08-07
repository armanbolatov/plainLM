#!/bin/bash
#SBATCH --job-name=qwen_ft
#SBATCH --partition=gpu
#SBATCH --account=eduard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/shared/home/arman.bolatov/plainLM/slurm_logs/%x_%A_%a.out
#SBATCH --error=/shared/home/arman.bolatov/plainLM/slurm_logs/%x_%A_%a.err
#SBATCH --array=0-7

# Same as slurm_ft2.sh but over the full 8-point LR grid in one array, for
# series added after the grid was extended. Subset with `sbatch --array=...`.
OPTIM=$1; TAG=$2; ALPHA=$3; SCALE=$4; MODEL=$5
LRS=(1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate plainLM
cd /shared/home/arman.bolatov/plainLM

python qwen_ft/finetune.py --optim $OPTIM --lr $LR \
  --alpha $ALPHA --scale-mult $SCALE --model $MODEL --tag $TAG
