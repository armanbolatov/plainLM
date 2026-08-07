#!/bin/bash
# Full-budget 70M LR sweep: one array task per point of the 11-point grid.
#
# Usage:
#   sbatch exps_70m/slurm_sweep.sh exps_70m/config/sweep_scion.yaml
#   sbatch --qos=guaranteed-eduard exps_70m/slurm_sweep.sh <config>
#   sbatch --array=10 exps_70m/slurm_sweep.sh <config>   # single LR (e.g. SF+)
#
#SBATCH --job-name=70m_sweep
#SBATCH --partition=gpu
#SBATCH --account=eduard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60000M
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/shared/home/arman.bolatov/plainLM/slurm_logs/%x_%A_%a.out
#SBATCH --error=/shared/home/arman.bolatov/plainLM/slurm_logs/%x_%A_%a.err
#SBATCH --array=0-10

LRS=(1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1e0)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
CONFIG=$1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate plainLM

cd /shared/home/arman.bolatov/plainLM

python sweep_run.py --config $CONFIG --lr $LR
