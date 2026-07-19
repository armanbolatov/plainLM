"""α sweep plot for G-NGN-Scion across all three scales.

Usage:
    python plotting/plot_alpha_sweep.py

Writes paper/figures/alpha_sweep.{png,pdf}. One figure, three subplots
(70M / 160M / 410M), gradient palette across α ∈ {1e-2, 3e-2, 5e-2, 7e-2, 1e-1}.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import wandb

from _style import apply_style, LW, S_LINE

SCALES = [
    ('70M',  ['steeldream/scion_70m_quarter_chinchilla'],  {}),
    ('160M', ['steeldream/scion_160m_quarter_chinchilla'], {}),
    ('410M', ['steeldream/scion_410m_quarter_chinchilla'], {}),
]

# Shared ylim across all subplots so scales compare directly.
YLIM = (2.7, 5.3)

# Sparse LR grid — one point per decade. Filter fetched data to just these,
# so the 410M panel doesn't show extra 3e-N points from the old 11-LR runs.
LR_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]

ALPHAS = [
    (1e-2, 'gngn_alpha_1e-2'),
    (3e-2, 'gngn_alpha_3e-2'),
    (5e-2, 'gngn_alpha_5e-2'),
    (7e-2, 'gngn_alpha_7e-2'),
    (1e-1, 'gngn_alpha_1e-1'),
]


def lr_from_name(name):
    try:
        return float(name.split('_lr_')[-1])
    except Exception:
        return None


def fetch(prefix, projects):
    api = wandb.Api()
    out = {}
    for proj in projects:
        for r in api.runs(proj, per_page=200):
            n = r.name or ''
            if not n.startswith(prefix + '_lr_') or r.state != 'finished':
                continue
            lr = lr_from_name(n)
            if lr is None:
                continue
            v = r.summary.get('valid/loss')
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                v = float('inf')
            if lr not in out or v < out[lr]:
                out[lr] = float(v)
    return out


def main():
    apply_style()
    cmap = mpl.colormaps['viridis']
    colors = [cmap(i / (len(ALPHAS) - 1)) for i in range(len(ALPHAS))]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True,
                             sharey=True)

    def closest_grid_lr(lr):
        """Snap a fetched LR to the nearest LR_GRID point (log-space), or None
        if it's not close enough (>10% off in log). Keeps 1e-3 but drops 3e-3."""
        for g in LR_GRID:
            if abs(math.log10(lr) - math.log10(g)) < 0.1:
                return g
        return None

    for ax, (title, projects, prefix_overrides) in zip(axes, SCALES):
        for (alpha, prefix), color in zip(ALPHAS, colors):
            prefix = prefix_overrides.get(alpha, prefix)
            d = fetch(prefix, projects)
            if not d:
                continue
            # Keep only points that snap onto LR_GRID
            keep = {}
            for lr, v in d.items():
                g = closest_grid_lr(lr)
                if g is not None and (g not in keep or v < keep[g]):
                    keep[g] = v
            if not keep:
                continue
            lrs = sorted(keep.keys())
            vals = [keep[lr] for lr in lrs]
            label = fr'$\alpha={alpha:g}$'
            ax.plot(lrs, vals, color=color, lw=LW, label=label)
            ax.scatter(lrs, vals, color=color, s=S_LINE,
                       edgecolor='white', linewidth=1.0)
        ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylim(*YLIM)
        ax.set_title(title)

    axes[0].set_ylabel('Validation loss')
    axes[0].legend(loc='upper left', framealpha=0.95)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, 'paper', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'alpha_sweep.png')
    plt.savefig(out, dpi=150)
    plt.savefig(out.replace('.png', '.pdf'))
    print(f'Saved {out} (+ .pdf)')


if __name__ == '__main__':
    main()
