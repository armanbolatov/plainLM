"""Alpha sweep for NGN-Scion on Qwen SFT — mirror of the pretraining figure.

Three panels (Qwen2.5-0.5B / 1.5B / 3B), viridis gradient over
alpha in {1e-2, 3e-2, 5e-2, 7e-2, 1e-1}, all with the tuned SFT scales
(8, 1, 8, 1024) and the 5-point LR grid. Reads local metrics.json.

    python plotting/plot_qwen_alpha_sweep.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt

from _style import apply_style, alpha_colors, minor_grid, shared_legend, LW, S_LINE

# alpha -> results tag per model prefix ('' / '15b_' / '3b_'). alpha=7e-2 is
# the headline gngn_p2 series; the others come from the dedicated sweep.
ALPHAS = [1e-2, 2e-2, 3e-2, 5e-2, 7e-2, 1e-1]

def tag_for(prefix, alpha):
    if alpha == 7e-2:
        return f'{prefix}gngn_p2'
    slug = f'{alpha:.0e}'.replace('e-0', 'e-')
    return f'{prefix}gngnp2_a{slug}'

PANELS = [('Qwen2.5-0.5B', ''), ('Qwen2.5-1.5B', '15b_'), ('Qwen2.5-3B', '3b_')]
RESULTS = 'qwen_ft/results_const'
YLIM = (1.1, 1.9)


def load(tag):
    out = {}
    for f in glob.glob(os.path.join(RESULTS, tag, 'lr_*', 'metrics.json')):
        d = json.load(open(f))
        v = d.get('final_val_loss')
        if isinstance(v, (int, float)) and v == v and v <= 10.0:
            out[d['lr']] = v
    return out


def main():
    apply_style()
    colors = alpha_colors(len(ALPHAS))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4),
                             constrained_layout=True, sharey=True)
    for ax, (title, prefix) in zip(axes, PANELS):
        for alpha, color in zip(ALPHAS, colors):
            d = load(tag_for(prefix, alpha))
            if not d:
                continue
            lrs = sorted(d)
            ax.plot(lrs, [d[l] for l in lrs], color=color, lw=LW,
                    label=fr'$\alpha={alpha:g}$')
            ax.scatter(lrs, [d[l] for l in lrs], color=color, s=S_LINE, zorder=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(*YLIM)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.set_xlabel('Learning rate')
        minor_grid(ax)
        ax.text(0.5, 0.85, title, transform=ax.transAxes,
                ha='center', va='top', fontsize=12, color='#2A2A2A')
    axes[0].set_ylabel('Validation loss')
    shared_legend(fig, axes[0])

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'paper', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'qwen_alpha_sweep.png')
    plt.savefig(out, dpi=150)
    plt.savefig(out.replace('.png', '.pdf'))
    print(f'Saved {out} (+ .pdf)')


if __name__ == '__main__':
    main()
