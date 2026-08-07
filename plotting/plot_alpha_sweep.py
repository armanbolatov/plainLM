"""α sweep for NGN-Scion and L-NGN-Scion across all three scales.

Usage:
    python plotting/plot_alpha_sweep.py

Writes paper/figures/alpha_sweep.{png,pdf} (G-NGN, 1x3).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt

from _data import fetch_on_grid
from _style import apply_style, alpha_colors, minor_grid, shared_legend, LW, S_LINE

SCALES = [
    ('70M',  'steeldream/scion_70m_alpha_chinchilla'),
    ('160M', 'steeldream/scion_160m_quarter_chinchilla'),
    ('410M', 'steeldream/scion_410m_quarter_chinchilla'),
]

ALPHAS = [1e-2, 3e-2, 5e-2, 7e-2, 1e-1]

VARIANTS = {
    'gngn': ('gngn_alpha_', 'NGN-Scion'),
    'lngn': ('lngn_alpha_', 'L-NGN-Scion'),
}

# One point per decade — keeps the panel readable and matches the sweep grid.
LR_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]

# L-NGN blows past the G-NGN range at high LR, so each row gets its own limits.
YLIM = {'gngn': (2.7, 5.3), 'lngn': (2.7, 9.0)}


def alpha_slug(a):
    """1e-2 -> '1e-2', 3e-2 -> '3e-2' (matches the exp_name in the configs)."""
    s = f'{a:.0e}'
    base, exp = s.split('e')
    return f'{base}e{int(exp)}'


def draw_row(axes, prefix, row_label, colors, annotate_scale, ylim):
    for ax, (scale, project) in zip(axes, SCALES):
        for alpha, color in zip(ALPHAS, colors):
            d = fetch_on_grid(f'{prefix}{alpha_slug(alpha)}_lr_', project, LR_GRID)
            if not d:
                continue
            lrs = sorted(d)
            vals = [d[lr] for lr in lrs]
            ax.plot(lrs, vals, color=color, lw=LW, label=fr'$\alpha={alpha:g}$')
            ax.scatter(lrs, vals, color=color, s=S_LINE, zorder=3)
        ax.set_xscale('log')
        ax.set_ylim(*ylim)
        ax.set_xlabel('Learning rate')
        minor_grid(ax)
        if annotate_scale:
            # Panel identity as in-axes text rather than a title.
            ax.text(0.5, 0.94, scale, transform=ax.transAxes,
                    ha='center', va='top', fontsize=12, color='#2A2A2A')
    axes[0].set_ylabel(f'Validation loss\n{row_label}' if row_label else 'Validation loss')


def main():
    apply_style()
    colors = alpha_colors(len(ALPHAS))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4),
                             constrained_layout=True, sharey=True)
    draw_row(axes, VARIANTS['gngn'][0], '', colors, annotate_scale=True,
             ylim=YLIM['gngn'])
    legend_src = axes[0]
    shared_legend(fig, legend_src)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, 'paper', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'alpha_sweep.png')
    plt.savefig(out, dpi=150)
    plt.savefig(out.replace('.png', '.pdf'))
    print(f'Saved {out} (+ .pdf)')


if __name__ == '__main__':
    main()
