#!/bin/bash
#SBATCH --job-name=qwen_ft
#SBATCH --partition=gpu
#SBATCH --account=eduard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/shared/home/arman.bolatov/plainLM/slurm_logs/%x_%j.out
#SBATCH --error=/shared/home/arman.bolatov/plainLM/slurm_logs/%x_%j.err
# Single run with an explicit LR (for off-grid points).
OPTIM=$1; TAG=$2; ALPHA=$3; SCALE=$4; MODEL=$5; LR=$6
source ~/miniconda3/etc/profile.d/conda.sh
conda activate plainLM
cd /shared/home/arman.bolatov/plainLM
python qwen_ft/finetune.py --optim $OPTIM --lr $LR --alpha $ALPHA --scale-mult $SCALE --model $MODEL --tag $TAG
