#!/bin/bash
#SBATCH --job-name=plainLM_prepdata
#SBATCH --account=eduard
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=12:00:00
#SBATCH --output=exps_410m/slurm_logs/prep_%j.out
#SBATCH --error=exps_410m/slurm_logs/prep_%j.err

set -euo pipefail

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export TRANSFORMERS_CACHE=/shared/models/huggingface/hub

source ~/miniconda3/etc/profile.d/conda.sh
conda activate plainLM
cd /shared/home/arman.bolatov/plainLM

# Re-chunk the already-tokenized dataset at seq_length=1024 so each row maps
# to exactly one training sequence (1024 input + 1 target = 1025 tokens/row).
# At seq_length=2048 we had 4.83M rows but the 410M sweep needs 502k micro-batches;
# at seq_length=1024 we have ~9.67M rows = 604k micro-batches, comfortably enough.
# The tokenized_dataset/ and tokenizer/ from the previous run are reused.

PYTHONPATH=. python data/datasets/prepare.py \
  --out_path="/shared/home/arman.bolatov/data/fwedu_10BT" \
  --cache_path="/shared/home/arman.bolatov/hf_datasets_cache" \
  --chunk \
  --dataset_path="HuggingFaceFW/fineweb-edu" \
  --dataset_split="train" \
  --dataset_name="sample-10BT" \
  --tokenizer="EleutherAI/gpt-neox-20b" \
  --seq_length=1024 \
  --split_train_valid=True \
  --n_tokens_valid=10000000
