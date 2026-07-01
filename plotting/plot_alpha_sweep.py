"""Generic α sweep plot for G-NGN-Scion across 70M / 160M / 410M.

Usage:
    python plotting/plot_alpha_sweep.py --scale 70m
    python plotting/plot_alpha_sweep.py --scale 160m
    python plotting/plot_alpha_sweep.py --scale 410m

Writes exps_{scale}/alpha_sweep.png. Compares α=3e-2, 5e-2, 7e-2.
"""
import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import wandb

SCALES = {
    '70m': dict(
        projects=['steeldream/scion_70m_simplify'],
        title='70M G-NGN-Scion: α (single global polyak multiplier) sweep — 5500 steps',
        ylim=(3.3, 6.0),
        out_dir='exps_70m',
    ),
    '160m': dict(
        projects=['steeldream/scion_160m'],
        title='160M G-NGN-Scion: α sweep — 12200 steps',
        ylim=(2.9, 6.0),
        out_dir='exps_160m',
    ),
    '410m': dict(
        # α=3e-2 / 5e-2 / 7e-2 sweep was done at 1/4 Chinchilla; 7e-2 also at full.
        projects=['steeldream/scion_410m_quarter_chinchilla', 'steeldream/scion_410m_full_chinchilla'],
        title='410M G-NGN-Scion: α sweep (1/4 Chinchilla for 3e-2 & 5e-2; full for 7e-2)',
        ylim=(2.6, 6.0),
        out_dir='exps_410m',
    ),
}

SOURCES = [
    ('α=3e-2', 'gngn_alpha_3e-2', 'C1', 's', '-'),
    ('α=5e-2', 'gngn_alpha_5e-2', 'C2', 'v', '-'),
    ('α=7e-2', 'gngn_alpha_7e-2', 'C3', '^', '-'),
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
            if not n.startswith(prefix + '_lr_'):
                continue
            if r.state != 'finished':
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
    p = argparse.ArgumentParser()
    p.add_argument('--scale', required=True, choices=list(SCALES.keys()))
    args = p.parse_args()
    cfg = SCALES[args.scale]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    print(f'{"alpha":<12} {"n":>4} {"peak":>8} {"±1 dec":>10} {"±2 dec":>10}')
    print('-' * 50)
    for label, prefix, color, marker, ls in SOURCES:
        d = fetch(prefix, cfg['projects'])
        finite = {k: v for k, v in d.items() if np.isfinite(v)}
        if not finite:
            print(f'{label:<12} (no data)')
            continue
        lr_opt = min(finite, key=finite.get)
        best = finite[lr_opt]
        s1 = max((v for lr, v in finite.items() if 0.1 <= lr / lr_opt <= 10), default=best) - best
        s2 = max((v for lr, v in finite.items() if 0.01 <= lr / lr_opt <= 100), default=best) - best
        print(f'{label:<12} {len(finite):>4} {best:>8.3f} {s1:>10.3f} {s2:>10.3f}')

        lrs = sorted(d.keys())
        vals = [d[lr] for lr in lrs]
        axes[0].plot(lrs, vals, color=color, marker=marker, linestyle=ls,
                     lw=2, ms=8, label=label)

        xs = [math.log10(lr / lr_opt) for lr in sorted(finite.keys())]
        ys = [finite[lr] for lr in sorted(finite.keys())]
        axes[1].plot(xs, ys, color=color, marker=marker, linestyle=ls, lw=2, ms=8,
                     label=f'{label}  (opt lr={lr_opt:.0e}, val={best:.3f})')

    for ax in axes:
        ax.set_yscale('log')
        ax.set_ylim(*cfg['ylim'])
        ax.grid(alpha=0.3, which='both')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('learning rate')
    axes[0].set_ylabel('val/loss')
    axes[0].set_title('Absolute LR')
    axes[0].legend(fontsize=10, loc='upper left')
    axes[1].axvline(0, color='black', alpha=0.3, ls='--')
    axes[1].set_xlabel(r'$\log_{10}(\mathrm{lr}/\mathrm{lr}_{\mathrm{opt}})$')
    axes[1].set_ylabel('val/loss')
    axes[1].set_title("Aligned around each method's optimum")
    axes[1].legend(fontsize=10, loc='upper left')

    plt.suptitle(cfg['title'], fontsize=12, y=1.0)
    plt.tight_layout()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(repo_root, cfg['out_dir'], 'alpha_sweep.png')
    plt.savefig(out, dpi=110, bbox_inches='tight')
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
