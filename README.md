# plainLM

Minimal, efficient causal-LM pretraining in PyTorch. Torch-compilable Transformer with RoPE, GLU, RMSNorm, DDP.

## Install

```bash
conda create --name plainLM python=3.12 -y && conda activate plainLM
pip install .
```

## Data

`data/datasets/prepare.py` downloads, tokenizes, and chunks any HF dataset. Example: `data/datasets/prepare_finewebedu_100BT.sh`.

## Single training run

```bash
python train.py --config=config/config.yaml
# multi-GPU:
torchrun --nproc_per_node=4 train.py --config=config/config.yaml
```

## Sweeps (per-scale experiments)

Each `exps_{70m,160m,410m}/` has the same layout:

```
config/          _base.yaml + one sweep_*.yaml per (method, α)
experiments/     metrics.json per (config × lr) run
*.png            lr_robustness, alpha_sweep, training_curves
```

Launch a single run in a sweep:

```bash
python sweep_run.py --config exps_410m/config/sweep_gngn_alpha_7e-2.yaml --lr 3e-4
```

`sweep_run.py` merges the config on top of the sibling `_base.yaml`. Set `divergence_threshold: 10.0` in a config to enable early-stop on catastrophic LRs.

## Regenerate plots

Three scripts, each takes `--scale`:

```bash
for scale in 70m 160m 410m; do
  python plotting/plot_lr_robustness.py   --scale $scale
  python plotting/plot_alpha_sweep.py     --scale $scale
  python plotting/plot_training_curves.py --scale $scale
done
```

They pull `valid/loss` from wandb; only `state == 'finished'` runs are used.

## Structure

```
plainLM/
├── cluster/       SLURM & Condor templates
├── config/        example training config
├── data/          data prep + streaming
├── engine/        train/eval step
├── models/        transformer
├── optim/         SCION + baselines
├── plotting/      per-scale plot scripts
├── exps_*/        per-scale configs, metrics, PNGs
├── archive/       old bayes sweeps
├── train.py       main training script
└── sweep_run.py   single-run launcher
```

## Citation

```bibtex
@misc{ajroldi2024plainlm,
  author = {Niccolò Ajroldi},
  title = {plainLM: Language Model Pretraining in PyTorch},
  year = {2024},
  howpublished = {\url{https://github.com/Niccolo-Ajroldi/plainLM}}
}
```

Inspired by [Cramming](https://github.com/JonasGeiping/cramming), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), [NanoGPT](https://github.com/karpathy/nanoGPT).
