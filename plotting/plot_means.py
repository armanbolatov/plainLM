"""
LR-sweep plots for the SCION-NGN means experiment (exps_means/).

Usage:
    python -m plotting.plot_means
"""
import os
from ._common import plt, load_runs, final_loss


EXP_ROOT = os.path.abspath('exps_means')
EXPERIMENTS_DIR = os.path.join(EXP_ROOT, 'experiments')
OUTPUT_DIR = os.path.join(EXP_ROOT, 'plots')

# (folder, label, color, marker, ls)
MEANS = [
    ('min',     'min',                  '#1f77b4', 'o', '-'),
    ('HM',      'HM (parallel)',        '#ff7f0e', 's', '-'),
    ('GM',      'GM',                   '#2ca02c', '^', '-'),
    ('AM_0.25', r'AM($\alpha$=0.25)',   '#d62728', 'D', '--'),
    ('AM_0.5',  r'AM($\alpha$=0.5)',    '#9467bd', 'D', '--'),
    ('AM_0.75', r'AM($\alpha$=0.75)',   '#8c564b', 'D', '--'),
    ('QM',      'QM',                   '#e377c2', 'v', ':'),
    ('max',     'max',                  '#17becf', 'x', ':'),
]


def plot_lr_sweep_loss():
    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    for folder, label, color, marker, ls in MEANS:
        runs = load_runs(EXPERIMENTS_DIR, folder)
        if not runs: continue
        lrs = [lr for lr, _ in runs]
        losses = [final_loss(m) for _, m in runs]
        ax.plot(lrs, losses, label=label, color=color,
                marker=marker, linestyle=ls, linewidth=1.8, markersize=7)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'Learning rate $\eta$', fontsize=12)
    ax.set_ylabel('Final train loss', fontsize=12)
    ax.set_title('LR sweep — SCION-NGN with different means combining $\\eta$ and the SCION Polyak step\n'
                 r'(small Transformer, 1000 steps, 16$\times$16 batch, 512 ctx)',
                 fontsize=12)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(3.8, 80)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'lr_sweep_means.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f'  saved {out}')


def plot_training_curves_best():
    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    for folder, label, color, _, ls in MEANS:
        runs = load_runs(EXPERIMENTS_DIR, folder)
        if not runs: continue
        best_lr, best_m = min(runs, key=lambda x: final_loss(x[1]))
        losses = best_m.get('train/loss_avg', best_m['train/loss'])
        ax.plot(best_m['step'], losses, label=f"{label} ($\\eta$={best_lr:.0e})",
                color=color, linestyle=ls, linewidth=1.6, alpha=0.9)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Train loss (EMA-smoothed)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Training curves at the best $\\eta$ for each mean', fontsize=12)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'training_curves_means.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f'  saved {out}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_lr_sweep_loss()
    plot_training_curves_best()
    print(f'\nAll plots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
