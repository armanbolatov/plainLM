"""Shared helpers for the plotting scripts."""
import json, glob, os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_runs(experiments_dir, method):
    """Return sorted list of (peak_lr, metrics_dict) for one method."""
    runs = []
    for path in sorted(glob.glob(f'{experiments_dir}/{method}/lr_*/metrics.json')):
        with open(path) as f:
            m = json.load(f)
        if not m or 'lr' not in m or 'train/loss' not in m:
            continue
        lr_val = m['lr']
        peak_lr = max(lr_val) if isinstance(lr_val, list) else lr_val
        runs.append((peak_lr, m))
    runs.sort(key=lambda x: x[0])
    return runs


def load_all(experiments_dir, methods):
    """Return dict[method] = list of (peak_lr, metrics_dict)."""
    return {m: load_runs(experiments_dir, m) for m in methods
            if load_runs(experiments_dir, m)}


def has_key(exps, method, key):
    if method not in exps or not exps[method]:
        return False
    return key in exps[method][0][1]


def final_loss(m):
    """Final loss: train/loss_avg if available, else last train/loss."""
    if 'train/loss_avg' in m and m['train/loss_avg']:
        return m['train/loss_avg'][-1]
    return m['train/loss'][-1]


def ema_smooth(values, alpha=0.05):
    """EMA smoothing pass, returns same-length list."""
    out = []
    val = values[0]
    for v in values:
        val = alpha * v + (1 - alpha) * val
        out.append(val)
    return out


def polyak_step(loss, dn_sq, eps=1e-12):
    """Return b_k = 2 f_k / ||g_k||_*^2 per step."""
    return [2.0 * fk / max(d, eps) for fk, d in zip(loss, dn_sq)]
