#!/bin/bash
#SBATCH --job-name=qwen_ft
#SBATCH --partition=gpu
#SBATCH --account=eduard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/shared/home/arman.bolatov/plainLM/slurm_logs/%x_%j.out
#SBATCH --error=/shared/home/arman.bolatov/plainLM/slurm_logs/%x_%j.err
TAG=$1; MODEL=$2; WD=$3; WARM=$4
source ~/miniconda3/etc/profile.d/conda.sh
conda activate plainLM
cd /shared/home/arman.bolatov/plainLM
python qwen_ft/finetune.py --optim sfplus --lr 1.0 --wd $WD --sfplus-warmup $WARM --model $MODEL --tag $TAG
