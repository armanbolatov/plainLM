#!/bin/bash
# Re-run a single (config, lr) point. Used to recover jobs that died from
# infrastructure failures (SLURM timeout, node loss) rather than divergence.
#
#   sbatch exps_410m/slurm_rerun.sh <config.yaml> <lr> [time]
#
# With divergence_threshold set at every scale, a genuinely divergent point
# now early-stops cleanly instead of crashing, so a rerun is conclusive either
# way: it either completes, or it finishes with no final loss (= diverged).
#SBATCH --job-name=rerun
#SBATCH --partition=gpu
#SBATCH --account=eduard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60000M
#SBATCH --time=30:00:00
#SBATCH --requeue
#SBATCH --output=/shared/home/arman.bolatov/plainLM/slurm_logs/%x_%j.out
#SBATCH --error=/shared/home/arman.bolatov/plainLM/slurm_logs/%x_%j.err

CONFIG=$1
LR=$2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate plainLM

cd /shared/home/arman.bolatov/plainLM

python sweep_run.py --config "$CONFIG" --lr "$LR"
