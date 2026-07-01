"""Lightweight checkpoint hooks.

Model checkpoints are intentionally not saved (large models × many sweep runs
would fill disk). Only `metrics.json` is written per run so downstream analysis
still works. `maybe_load_checkpoint` is kept as a stub for API symmetry with
`train.py`; resuming from disk is not supported by this repo.
"""
import json
import os

import utils


def save_checkpoint(step, model, engine, cfg, metrics):
    exp_dir = utils.get_exp_dir_path(cfg)
    metrics_path = os.path.join(exp_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(dict(metrics), f)


def maybe_load_checkpoint(cfg):
    return None
