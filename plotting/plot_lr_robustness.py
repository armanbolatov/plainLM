"""Generic LR robustness plot for 70M / 160M / 410M.

Usage:
    python plotting/plot_lr_robustness.py --scale 70m
    python plotting/plot_lr_robustness.py --scale 160m
    python plotting/plot_lr_robustness.py --scale 410m

Writes exps_{scale}/lr_robustness.png. Pulls from wandb.
Only includes runs with state='finished' (skips currently running / crashed).
"""
import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import wandb
from collections import defaultdict

SCALES = {
    '70m': dict(
        projects=['steeldream/scion_70m_final', 'steeldream/scion_70m_simplify'],
        title='70M Chinchilla LR robustness — 5500 steps',
        ylim=(3.3, 6.0),
        out_dir='exps_70m',
    ),
    '160m': dict(
        projects=['steeldream/scion_160m'],
        title='160M Chinchilla LR robustness — 12200 steps = 3.2B tokens = 20 TPP',
        ylim=(2.9, 6.0),
        out_dir='exps_160m',
    ),
    '410m': dict(
        projects=['steeldream/scion_410m_full_chinchilla'],
        title='410M Full Chinchilla LR robustness — 15680 steps = 8.2B tokens = 20 TPP',
        ylim=(2.6, 6.0),
        out_dir='exps_410m',
    ),
}

METHODS = [
    ('Scion',                   'scion',           'C0', 'd', '-'),
    ('L-NGN-Scion',             'lngn_scion',      'C1', 'D', '-'),
    ('G-NGN-Scion (α=7e-2)',    'gngn_alpha_7e-2', 'C2', 'v', '-'),
    ('AdamW (β=.9/.95)',        'adamw',           'C3', 'P', '-'),
    ('muonmax_momo',            'muonmax_momo',    'C4', 'X', '-'),
    ('NGN-MDv1',                'ngnmdv1',         'C5', 'h', '-'),
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
            if '_lr_' not in n:
                continue
            if r.state != 'finished':
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

    data = fetch(cfg['projects'])
    print('Run counts by method:')
    for _, prefix, _, _, _ in METHODS:
        print(f'  {prefix:18s} {len(data.get(prefix, {}))} LRs')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for label, prefix, color, marker, ls in METHODS:
        d = data.get(prefix, {})
        if not d:
            continue
        lrs = sorted(d.keys())
        vals = [d[lr] for lr in lrs]
        ax.plot(lrs, vals, color=color, marker=marker, linestyle=ls,
                linewidth=2, markersize=8, label=label)
    ax.set_xscale('log')
    ax.set_xlabel('learning rate')
    ax.set_ylabel('val/loss')
    ax.set_title('Absolute LR')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3, which='both')
    ax.set_yscale('log')
    ax.set_ylim(*cfg['ylim'])

    ax = axes[1]
    for label, prefix, color, marker, ls in METHODS:
        d = data.get(prefix, {})
        finite = {lr: v for lr, v in d.items() if np.isfinite(v)}
        if not finite:
            continue
        lr_opt = min(finite, key=finite.get)
        xs, ys = [], []
        for lr in sorted(d.keys()):
            v = d[lr]
            if np.isfinite(v):
                xs.append(math.log10(lr / lr_opt))
                ys.append(v)
        ax.plot(xs, ys, color=color, marker=marker, linestyle=ls,
                linewidth=2, markersize=8,
                label=f'{label}  (opt lr={lr_opt:.0e}, val={finite[lr_opt]:.3f})')
    ax.axvline(0, color='black', alpha=0.3, linestyle='--')
    ax.set_xlabel(r'$\log_{10}(\mathrm{lr} / \mathrm{lr}_{\mathrm{opt}})$')
    ax.set_ylabel('val/loss')
    ax.set_title("Aligned around each method's optimum")
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim(*cfg['ylim'])

    plt.suptitle(cfg['title'], fontsize=13, y=1.0)
    plt.tight_layout()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(repo_root, cfg['out_dir'], 'lr_robustness.png')
    plt.savefig(out, dpi=110, bbox_inches='tight')
    print(f'\nSaved {out}')

    print('\nSpread analysis (val/loss - best):')
    print(f'{"method":<24} {"n":>3} {"peak":>7} {"opt lr":>10} {"±1 dec":>9} {"±2 dec":>9}')
    print('-' * 70)
    for label, prefix, _, _, _ in METHODS:
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
