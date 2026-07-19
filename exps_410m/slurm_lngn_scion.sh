#!/bin/bash
#SBATCH --job-name=410m_lngn_scion
#SBATCH --partition=gpu
#SBATCH --account=eduard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60000M
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/shared/home/arman.bolatov/plainLM/slurm_logs/%x_%A_%a.out
#SBATCH --error=/shared/home/arman.bolatov/plainLM/slurm_logs/%x_%A_%a.err
#SBATCH --array=0-10

LRS=(1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1e0)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate plainLM

cd /shared/home/arman.bolatov/plainLM

python sweep_run.py \
  --config exps_410m/config/sweep_lngn_scion.yaml \
  --lr $LR
