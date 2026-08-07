"""SFT training curves: are the flat-LR runs actually different runs?

Two panels (Qwen2.5-0.5B): NGN-Scion and AdamW, validation loss vs training
step, one curve per LR (viridis by LR). Answers the "are the final losses
identical / is there a bug" question: G-NGN trajectories differ early (the cap
is inactive at small LR) and converge to close-but-distinct finals; AdamW fans
out by decades. Reads the per-run history stored in metrics.json.

    python plotting/plot_qwen_curves.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt

from _style import apply_style, alpha_colors, minor_grid, LW

# Two alphas side by side: the headline 7e-2 and the tight 2e-2. The second
# panel shows the small-alpha runs are still descending at step 600 rather than
# stuck, so the alpha-sweep plateau is undertraining and not a failure mode.
PANELS = [
    (r'$\alpha=7\times10^{-2}$', 'gngn_p2',        (1.29, 1.52)),
    (r'$\alpha=2\times10^{-2}$', 'gngnp2_a2e-2',   (1.29, 1.95)),
]
LRS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
RESULTS = 'qwen_ft/results_const'


def load_history(tag, lr):
    slug = f'{lr:.0e}'.replace('e-0', 'e-').replace('e+0', 'e')
    f = os.path.join(RESULTS, tag, f'lr_{slug}', 'metrics.json')
    if not os.path.exists(f):
        return None
    h = json.load(open(f)).get('history', [])
    steps = [r['step'] for r in h]
    vals = [r['val_loss'] for r in h]
    return steps, vals


def main():
    apply_style()
    colors = alpha_colors(len(LRS))
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.5 * len(PANELS), 4.5),
                             constrained_layout=True)
    axes = [axes] if len(PANELS) == 1 else list(axes)
    for ax, (title, tag, ylim) in zip(axes, PANELS):
        for lr, color in zip(LRS, colors):
            h = load_history(tag, lr)
            if h is None:
                continue
            steps, vals = h
            ax.plot(steps, vals, color=color, lw=LW,
                    label=fr'$\eta={lr:g}$')
            final = vals[-1]
            ax.annotate(f'{final:.3f}', (steps[-1], final), fontsize=8,
                        color=color, xytext=(4, 0), textcoords='offset points')
        ax.set_yscale('log')
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.set_xlabel('Training step')
        minor_grid(ax, axis='y')
        ax.text(0.5, 0.94, title, transform=ax.transAxes,
                ha='center', va='top', fontsize=11, color='#2A2A2A')
    axes[0].set_ylabel('Validation loss')
    axes[0].legend(loc='upper right', fontsize=9, frameon=False)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'paper', 'figures')
    out = os.path.join(out_dir, 'qwen_curves.png')
    plt.savefig(out, dpi=150)
    plt.savefig(out.replace('.png', '.pdf'))
    print(f'Saved {out} (+ .pdf)')


if __name__ == '__main__':
    main()
