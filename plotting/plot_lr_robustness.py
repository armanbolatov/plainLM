"""LR robustness across 70M / 160M / 410M — one figure, three panels.

Usage:
    python plotting/plot_lr_robustness.py             # absolute LR on the x-axis
    python plotting/plot_lr_robustness.py --aligned   # x = log10(eta / eta*)

Writes paper/figures/lr_robustness{,_aligned}.{png,pdf}. Pulls finished runs
from wandb.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt

from _data import fetch_by_prefix, LOSS_CEILING
from _style import apply_style, minor_grid, shared_legend, PALETTE, LW, S_LINE

SCALES = [
    ('70M',  ['steeldream/scion_70m_chinchilla']),
    ('160M', ['steeldream/scion_160m']),
    ('410M', ['steeldream/scion_410m_full_chinchilla']),
]

METHODS = [
    ('Scion',                                      'scion',           'o'),
    ('NGN-Scion',                                'gngn_alpha_7e-2', 'v'),
    ('AdamW',                                      'adamw',           'P'),
    ('Muonmax-Momo',                               'muonmax_momo',    'X'),
    ('NGN-MDv1',                                   'ngnmdv1',         'h'),
    ('ScheduleFree+',                              'sfplus_wd',       'D'),
]

YLIM = (2.6, LOSS_CEILING)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--aligned', action='store_true',
                   help="x-axis becomes log10(eta / eta*) per method")
    args = p.parse_args()

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4),
                             constrained_layout=True, sharey=True)

    for ax, (scale, projects) in zip(axes, SCALES):
        data = fetch_by_prefix(projects)
        for label, prefix, marker in METHODS:
            d = data.get(prefix, {})
            if not d:
                continue
            color = PALETTE[prefix]
            if prefix.startswith('sfplus'):
                # LR-free: single run, drawn as a horizontal reference line.
                ax.axhline(list(d.values())[0], color=color, ls=':', lw=LW,
                           label=label)
                continue
            if args.aligned:
                lr_opt = min(d, key=d.get)
                xs = [math.log10(lr / lr_opt) for lr in sorted(d)]
            else:
                xs = sorted(d)
            ys = [d[lr] for lr in sorted(d)]
            ax.plot(xs, ys, color=color, lw=LW, label=label)
            ax.scatter(xs, ys, color=color, marker=marker, s=S_LINE, zorder=3)

        if args.aligned:
            ax.axvline(0, color='white', lw=1.5, zorder=1)
            ax.set_xlabel(r'$\log_{10}(\eta / \eta^\star)$')
            ax.set_xlim(-2.5, 2.5)
        else:
            ax.set_xscale('log')
            ax.set_xlabel('Learning rate')
        ax.set_yscale('log')
        ax.set_ylim(*YLIM)
        minor_grid(ax, axis='y')
        ax.text(0.5, 0.94, scale, transform=ax.transAxes,
                ha='center', va='top', fontsize=12, color='#2A2A2A')

    axes[0].set_ylabel('Validation loss')
    shared_legend(fig, axes[0])

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, 'paper', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    name = 'lr_robustness_aligned' if args.aligned else 'lr_robustness'
    out = os.path.join(out_dir, f'{name}.png')
    plt.savefig(out, dpi=150)
    plt.savefig(out.replace('.png', '.pdf'))
    print(f'Saved {out} (+ .pdf)')


if __name__ == '__main__':
    main()
