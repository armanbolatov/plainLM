"""Training curves at each method's best LR — one figure, three panels.

Usage:
    python plotting/plot_training_curves.py

Writes paper/figures/training_curves.{png,pdf}.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
import wandb

from _style import apply_style, minor_grid, shared_legend, PALETTE, LW

SCALES = [
    ('70M',  ['steeldream/scion_70m_chinchilla'],  5500),
    ('160M', ['steeldream/scion_160m'],                                       12200),
    ('410M', ['steeldream/scion_410m_full_chinchilla'],                       15680),
]

METHODS = [
    ('Scion',                                      'scion'),
    ('NGN-Scion', 'gngn_alpha_7e-2'),
    ('AdamW',                                      'adamw'),
    ('Muonmax-Momo',                               'muonmax_momo'),
    ('NGN-MDv1',                                   'ngnmdv1'),
]

YLIM = (2.6, 5.0)


def find_best_run(prefix, projects):
    api = wandb.Api()
    best_run, best_val, best_lr = None, float('inf'), None
    for proj in projects:
        for r in api.runs(proj, per_page=200):
            n = r.name or ''
            if not n.startswith(prefix + '_lr_') or r.state != 'finished':
                continue
            try:
                lr = float(n.split('_lr_')[-1])
            except ValueError:
                continue
            v = r.summary.get('valid/loss')
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                continue
            if v < best_val:
                best_val, best_run, best_lr = v, r, lr
    return best_run, best_val, best_lr


def fetch_history(run, key='valid/loss', step_key='step'):
    steps, vals = [], []
    for row in run.scan_history(keys=[step_key, key], page_size=2000):
        s, v = row.get(step_key), row.get(key)
        if s is None or v is None or not np.isfinite(v):
            continue
        steps.append(s)
        vals.append(v)
    return np.array(steps), np.array(vals)


def main():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4),
                             constrained_layout=True, sharey=True)

    for ax, (scale, projects, max_step) in zip(axes, SCALES):
        for label, prefix in METHODS:
            run, val, lr = find_best_run(prefix, projects)
            if run is None:
                continue
            steps, vals = fetch_history(run)
            if len(steps) == 0:
                continue
            ax.plot(steps, vals, color=PALETTE[prefix], lw=LW, label=label)
        ax.set_xlim(0, max_step)
        ax.set_yscale('log')
        ax.set_ylim(*YLIM)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.set_xlabel('Training step')
        minor_grid(ax)
        ax.text(0.5, 0.94, scale, transform=ax.transAxes,
                ha='center', va='top', fontsize=12, color='#2A2A2A')

    axes[0].set_ylabel('Validation loss')
    shared_legend(fig, axes[0])

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, 'paper', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(out, dpi=150)
    plt.savefig(out.replace('.png', '.pdf'))
    print(f'Saved {out} (+ .pdf)')


if __name__ == '__main__':
    main()
