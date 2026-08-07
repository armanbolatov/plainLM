"""Shared wandb fetching for the figure scripts.

One place defines what counts as a usable data point, so every figure applies
the same rule:

  * only 'finished' runs are used;
  * a final valid/loss above LOSS_CEILING is dropped.

Every scale sets `divergence_threshold: 10.0`, so a diverging run early-stops
and finishes without a final loss. A curve that ends early therefore means the
method diverged there — not that a job was lost. Jobs that died for
infrastructure reasons are re-launched rather than plotted.
"""
import math
from collections import defaultdict

import numpy as np
import wandb

LOSS_CEILING = 10.0


def _lr_of(name):
    try:
        return float(name.split('_lr_')[-1])
    except ValueError:
        return None


def _usable(run):
    """Final valid/loss if the run completed and converged, else None."""
    if run.state != 'finished':
        return None
    v = run.summary.get('valid/loss')
    if not isinstance(v, (int, float)) or not np.isfinite(v) or v > LOSS_CEILING:
        return None
    return float(v)


def fetch_by_prefix(projects):
    """{run_prefix: {lr: valid/loss}} across one or more wandb projects."""
    api = wandb.Api()
    out = defaultdict(dict)
    for proj in projects:
        for r in api.runs(proj, per_page=200):
            n = r.name or ''
            if '_lr_' not in n:
                continue
            lr, v = _lr_of(n), _usable(r)
            if lr is None or v is None:
                continue
            prefix = n.split('_lr_')[0]
            if lr not in out[prefix] or v < out[prefix][lr]:
                out[prefix][lr] = v
    return out


def snap_to_grid(lr, grid, tol=0.1):
    """Nearest grid point in log space, or None if further than `tol` decades."""
    for g in grid:
        if abs(math.log10(lr) - math.log10(g)) < tol:
            return g
    return None


def fetch_on_grid(prefix, project, grid):
    """{grid_lr: valid/loss} for runs named `prefix` + lr, snapped onto `grid`."""
    api = wandb.Api()
    out = {}
    for r in api.runs(project, per_page=200):
        n = r.name or ''
        if not n.startswith(prefix) or '_lr_' not in n:
            continue
        lr, v = _lr_of(n), _usable(r)
        if lr is None or v is None:
            continue
        g = snap_to_grid(lr, grid)
        if g is None:
            continue
        if g not in out or v < out[g]:
            out[g] = v
    return out
