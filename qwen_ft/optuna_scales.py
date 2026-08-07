"""Optuna search over G-NGN-Scion per-group scales (+ alpha) for Qwen SFT.

Goal: close the peak-loss gap to AdamW. Each trial launches finetune.py as a
subprocess on the same GPU (fresh CUDA state) at a fixed mid-plateau LR — the
cap binds there, so the result measures the (scales, alpha) quality, not LR.

    python qwen_ft/optuna_scales.py --trials 40

Resumable: state lives in qwen_ft/optuna_scales.db.
"""
import argparse
import json
import os
import subprocess
import sys

import optuna

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LR = 3e-4                       # mid-plateau; cap binds


def run_trial(trial):
    m = trial.suggest_float('matrix', 0.5, 16.0, log=True)
    o = trial.suggest_float('oned', 1.0, 64.0, log=True)
    e = trial.suggest_float('embed', 8.0, 512.0, log=True)
    h = trial.suggest_float('lm_head', 256.0, 32768.0, log=True)
    a = trial.suggest_float('alpha', 5e-3, 1e-1, log=True)
    tag = f'optuna_t{trial.number}'
    cmd = [sys.executable, os.path.join(HERE, 'finetune.py'),
           '--optim', 'gngn', '--lr', str(LR), '--alpha', str(a),
           '--scales', f'{m},{o},{e},{h}', '--tag', tag]
    subprocess.run(cmd, cwd=ROOT, check=True)
    # finetune.py writes to results_const (its --out default). Raw outputs of
    # the historical optuna_t0..t39 trials live in archive/qwen_ft_results_cosine.
    mfile = os.path.join(ROOT, 'qwen_ft', 'results_const', tag, 'lr_3e-4', 'metrics.json')
    val = json.load(open(mfile))['final_val_loss']
    return val


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=40)
    args = p.parse_args()

    study = optuna.create_study(
        study_name='gngn_scales',
        storage=f'sqlite:///{os.path.join(HERE, "optuna_scales.db")}',
        load_if_exists=True,
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    if len(study.trials) == 0:
        # Seed points: current defaults, and two literature-inspired guesses
        # (Moonlight-style RMS matching suggests larger matrix / smaller head).
        study.enqueue_trial({'matrix': 4.0, 'oned': 8.0, 'embed': 64.0,
                             'lm_head': 4096.0, 'alpha': 2e-2})
        study.enqueue_trial({'matrix': 8.0, 'oned': 8.0, 'embed': 32.0,
                             'lm_head': 1024.0, 'alpha': 3e-2})
        study.enqueue_trial({'matrix': 2.0, 'oned': 4.0, 'embed': 128.0,
                             'lm_head': 8192.0, 'alpha': 1e-2})
    study.optimize(run_trial, n_trials=args.trials)
    print('BEST:', study.best_value, study.best_params)


if __name__ == '__main__':
    main()
