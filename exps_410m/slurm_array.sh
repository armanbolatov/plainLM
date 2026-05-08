#!/bin/bash
# -----------------------------------------------------------------------------
# SLURM job-array launcher for the 410M sweep on the Martin cluster.
#
# Topology: 8 sweep configs (one per method) x 8 LRs = 64 runs.
# Each array task runs ONE (method, lr) pair on a single GPU.
#
# Submit with:
#   sbatch exps_410m/slurm_array.sh
#
# To launch only a subset, restrict the array range, e.g.:
#   sbatch --array=0-7 exps_410m/slurm_array.sh          # all LRs for first method
#   sbatch --array=0,8,16,24,32,40,48,56 ...             # first LR for every method
#
# To burst beyond your group's quota using idle GPUs (preemptable):
#   sbatch --qos=normal exps_410m/slurm_array.sh
# -----------------------------------------------------------------------------

#SBATCH --job-name=plainLM_410m
#SBATCH --account=eduard                                # <-- change to your group
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=125000
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --array=0-79%8                                  # 80 jobs, 8 concurrently
#SBATCH --output=exps_410m/slurm_logs/%x_%A_%a.out
#SBATCH --error=exps_410m/slurm_logs/%x_%A_%a.err

set -euo pipefail

# --- Environment -------------------------------------------------------------
# Shared HuggingFace cache (keeps tokenizers/models off your 500 GB home quota)
export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export TRANSFORMERS_CACHE=/shared/models/huggingface/hub

# Conda env (expects `plainLM` env from README: conda create -n plainLM python=3.12)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate plainLM

# torch.compile cache on local scratch to avoid NFS thrash
export TORCHINDUCTOR_CACHE_DIR=/scratch/${USER}/torch_inductor
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

cd /shared/home/arman.bolatov/plainLM

# --- Sweep grid --------------------------------------------------------------
# The order here defines the mapping SLURM_ARRAY_TASK_ID -> (config, lr).
CONFIGS=(
  sweep_adamw.yaml
  sweep_L1.yaml
  sweep_L1_unc.yaml
  sweep_min.yaml
  sweep_muonmax_momo.yaml
  sweep_ngnmdv1.yaml
  sweep_polyak.yaml
  sweep_standard.yaml
)
LRS=(1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1)

N_LRS=${#LRS[@]}
cfg_idx=$(( SLURM_ARRAY_TASK_ID / N_LRS ))
lr_idx=$(( SLURM_ARRAY_TASK_ID % N_LRS ))

config_file="exps_410m/config/${CONFIGS[$cfg_idx]}"
lr="${LRS[$lr_idx]}"

echo "=== Task $SLURM_ARRAY_TASK_ID ==="
echo "config = $config_file"
echo "lr     = $lr"
echo "host   = $(hostname)"
echo "gpu    = $CUDA_VISIBLE_DEVICES"

# --- Launch ------------------------------------------------------------------
# sweep_run.py handles _base.yaml merge, metrics.json skip-if-done, wandb naming.
python sweep_run.py --config "$config_file" --lr "$lr"
