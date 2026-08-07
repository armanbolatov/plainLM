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
slurm_*.sh       sbatch array launchers (one task per LR)
```

Launch a single run in a sweep:

```bash
python sweep_run.py --config exps_410m/config/sweep_gngn_alpha_7e-2.yaml --lr 3e-4
```

`sweep_run.py` merges the config on top of the sibling `_base.yaml`. Set `divergence_threshold: 10.0` in a config to enable early-stop on catastrophic LRs.

## Regenerate plots

Each script writes one multi-panel figure (70M / 160M / 410M side by side) to
`paper/figures/` as both `.png` and `.pdf`:

```bash
python plotting/plot_lr_robustness.py                  # lr_robustness
python plotting/plot_lr_robustness.py --aligned        # lr_robustness_aligned
python plotting/plot_alpha_sweep.py                    # alpha_sweep (G-NGN)
python plotting/plot_training_curves.py                # training_curves
python plotting/plot_proximal_stepsize.py              # proximal_stepsize
python plotting/plot_qwen_ft.py                        # qwen_ft (SFT, local)
python plotting/plot_qwen_curves.py                    # qwen_curves (SFT, local)
python plotting/plot_qwen_alpha_sweep.py               # qwen_alpha_sweep (SFT, local)
```

The first four pull `valid/loss` from wandb; only `state == 'finished'` runs
are used. The `plot_qwen_*` scripts read local metrics from
`qwen_ft/results_const/`. `plot_proximal_stepsize.py` is self-contained
(numpy only) and reproduces the proximal-model figure; derivations are in
[docs/proximal_derivations.md](docs/proximal_derivations.md).

Shared styling lives in `plotting/_style.py`.

## Qwen SFT benchmark

`qwen_ft/finetune.py` fine-tunes Qwen2.5-{0.5B,1.5B,3B} on Alpaca with a
constant LR (600 steps, bs 8), one (optimizer, lr) per invocation. Optimizers:
`gngn`, `scion`, `adamw`, `muonmax`, `sfplus`, `ngnmdv1`, and more — see
`--help`. G-NGN-Scion uses α=7e-2 with SFT-tuned LMO scales `8,1,8,1024`
(found by `qwen_ft/optuna_scales.py`).

```bash
# full 8-point LR grid for one (optimizer, model):
sbatch qwen_ft/slurm_ft_full.sh adamw 15b_adamw - - Qwen/Qwen2.5-1.5B
# same but with explicit LMO scales (alpha-sweep series):
sbatch qwen_ft/slurm_scales_full.sh gngn gngn_p2 7e-2 "8,1,8,1024" Qwen/Qwen2.5-0.5B
```

Results land in `qwen_ft/results_const/<tag>/lr_<lr>/metrics.json`, which the
three `plot_qwen_*.py` scripts consume directly.

## Structure

```
plainLM/
├── cluster/       SLURM & Condor templates
├── config/        example training config
├── data/          data prep + streaming
├── engine/        train/eval step
├── models/        transformer
├── optim/         SCION + baselines
├── plotting/      figure scripts + shared style
├── docs/          proximal-model derivations
├── exps_*/        per-scale configs, metrics, launchers
├── qwen_ft/       Qwen SFT benchmark (trainer, launchers, results)
├── paper/         LaTeX source + generated figures
├── archive/       retired one-off drivers, superseded launchers, old sweeps
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
