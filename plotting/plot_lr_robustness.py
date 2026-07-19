"""Generic LR robustness plot for 70M / 160M / 410M.

Usage:
    python plotting/plot_lr_robustness.py --scale 70m
    python plotting/plot_lr_robustness.py --scale 160m
    python plotting/plot_lr_robustness.py --scale 410m

Writes exps_{scale}/lr_robustness.{png,pdf}. Pulls from wandb, finished runs
only. Style: seaborn whitegrid + default fonts + thick lines (see _style.py).
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
import wandb
from collections import defaultdict

from _style import apply_style, PALETTE, LW, S_LINE

SCALES = {
    '70m': dict(
        projects=['steeldream/scion_70m_final', 'steeldream/scion_70m_simplify'],
        title='70M Chinchilla',
        ylim=(3.3, 10.0),
        out_dir='exps_70m',
    ),
    '160m': dict(
        projects=['steeldream/scion_160m'],
        title='160M Chinchilla',
        ylim=(2.9, 10.0),
        out_dir='exps_160m',
    ),
    '410m': dict(
        projects=['steeldream/scion_410m_full_chinchilla'],
        title='410M Chinchilla',
        ylim=(2.6, 10.0),
        out_dir='exps_410m',
    ),
}

METHODS = [
    ('Scion',                         'scion',           'o'),
    ('L-NGN-Scion',                   'lngn_scion',      'D'),
    (r'G-NGN-Scion ($\alpha=7\!\times\!10^{-2}$)', 'gngn_alpha_7e-2', 'v'),
    ('AdamW',                         'adamw',           'P'),
    ('Muonmax-Momo',                  'muonmax_momo',    'X'),
    ('NGN-MDv1',                      'ngnmdv1',         'h'),
]


def lr_from_name(name):
    try:
        return float(name.split('_lr_')[-1])
    except Exception:
        return None


def fetch(projects):
    api = wandb.Api()
    out = defaultdict(dict)
    for proj in projects:
        for r in api.runs(proj, per_page=200):
            n = r.name or ''
            if '_lr_' not in n or r.state != 'finished':
                continue
            prefix = n.split('_lr_')[0]
            lr = lr_from_name(n)
            if lr is None:
                continue
            v = r.summary.get('valid/loss')
            if v is None or not np.isfinite(v):
                v = float('inf')
            if lr not in out[prefix] or v < out[prefix][lr]:
                out[prefix][lr] = float(v)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scale', required=True, choices=list(SCALES.keys()))
    args = p.parse_args()
    cfg = SCALES[args.scale]

    apply_style()

    data = fetch(cfg['projects'])
    print('Run counts by method:')
    for _, prefix, _ in METHODS:
        print(f'  {prefix:18s} {len(data.get(prefix, {}))} LRs')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    ax = axes[0]
    for label, prefix, marker in METHODS:
        d = data.get(prefix, {})
        if not d:
            continue
        color = PALETTE[prefix]
        lrs = sorted(d.keys())
        vals = [d[lr] for lr in lrs]
        ax.plot(lrs, vals, color=color, lw=LW, label=label, zorder=3)
        ax.scatter(lrs, vals, color=color, marker=marker, s=S_LINE, zorder=4,
                   edgecolor='white', linewidth=1.0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Validation loss')
    ax.set_ylim(*cfg['ylim'])

    ax = axes[1]
    for label, prefix, marker in METHODS:
        d = data.get(prefix, {})
        finite = {lr: v for lr, v in d.items() if np.isfinite(v)}
        if not finite:
            continue
        color = PALETTE[prefix]
        lr_opt = min(finite, key=finite.get)
        xs, ys = [], []
        for lr in sorted(d.keys()):
            v = d[lr]
            if np.isfinite(v):
                xs.append(math.log10(lr / lr_opt))
                ys.append(v)
        ax.plot(xs, ys, color=color, lw=LW, label=label, zorder=3)
        ax.scatter(xs, ys, color=color, marker=marker, s=S_LINE, zorder=4,
                   edgecolor='white', linewidth=1.0)
    ax.axvline(0, color='black', alpha=0.3, ls='--', lw=1.2)
    ax.set_yscale('log')
    ax.set_xlabel(r'$\log_{10}(\eta / \eta^\star)$')
    ax.set_ylim(*cfg['ylim'])
    ax.set_xlim(-2.5, 2.5)

    axes[0].legend(loc='upper left', framealpha=0.95)
    fig.suptitle(cfg['title'])

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(repo_root, cfg['out_dir'], 'lr_robustness.png')
    plt.savefig(out, dpi=150)
    plt.savefig(out.replace('.png', '.pdf'))
    print(f'\nSaved {out} (+ .pdf)')

    print('\nSpread analysis (val/loss - best):')
    print(f'{"method":<24} {"n":>3} {"peak":>7} {"opt lr":>10} {"±1 dec":>9} {"±2 dec":>9}')
    print('-' * 70)
    for label, prefix, _ in METHODS:
        d = data.get(prefix, {})
        finite = {lr: v for lr, v in d.items() if np.isfinite(v)}
        if not finite:
            print(f'{label:<24} (no data)')
            continue
        lr_opt = min(finite, key=finite.get)
        best = finite[lr_opt]
        s1 = max((v for lr, v in finite.items() if 0.1 <= lr / lr_opt <= 10), default=best) - best
        s2 = max((v for lr, v in finite.items() if 0.01 <= lr / lr_opt <= 100), default=best) - best
        print(f'{label:<24} {len(finite):>3} {best:>7.3f} {lr_opt:>10.0e} {s1:>9.3f} {s2:>9.3f}')


if __name__ == '__main__':
    main()
