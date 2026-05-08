"""
Main LR-sweep comparison plots across optimizers (70m / 160m).

Reads {exp_root}/experiments/{method}/lr_*/metrics.json and writes 7 plots
to {exp_root}/plots/. The output files are byte-compatible with the previous
plot_sweep.py script.

Usage:
    python -m plotting.plot_sweep [exp_root]        # default: exps_70m
"""
import os, sys
import numpy as np
from ._common import (plt, load_all, has_key, final_loss, ema_smooth)


SWEEPS = {
    'L1':           {'label': 'SCION HM (constr)',    'color': '#9467bd', 'marker': 'D', 'ls': '--'},
    'L1_unc':       {'label': 'SCION HM (unconstr)',  'color': '#8c564b', 'marker': 'P', 'ls': '--'},
    'min':          {'label': 'SCION min (constr)',   'color': '#2ca02c', 'marker': '^', 'ls': '-'},
    'polyak':       {'label': 'SCION polyak (constr)','color': '#d62728', 'marker': 'v', 'ls': '-'},
    'standard':     {'label': 'Standard SCION',       'color': '#000000', 'marker': 'x', 'ls': ':'},
    'adamw':        {'label': 'AdamW',                'color': '#e377c2', 'marker': '*', 'ls': '-'},
    'ngnmdv1':      {'label': 'NGN-MDv1',             'color': '#17becf', 'marker': 'h', 'ls': '-'},
    'muonmax_momo': {'label': 'MuonMax-Momo',         'color': '#bcbd22', 'marker': 'd', 'ls': '-'},
}
HORIZONTAL_METHODS = {'polyak'}  # ignore user LR, draw as horizontal band


def _savefig(out_dir, fname):
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    print(f'  saved {fname}')


def _plot_vs_lr(exps, out_dir, fname, title, ylabel, metric_key, use_ylog=True,
                extra_diag=False, ymin=None, ymax=None):
    """Generic 'y vs nominal LR, one point per run at mid-training' plot."""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    for method, cfg in SWEEPS.items():
        if method not in exps or not has_key(exps, method, metric_key):
            continue
        lrs = [lr for lr, _ in exps[method]]
        mids = [len(m['step']) // 2 for _, m in exps[method]]
        ys = [m[metric_key][i] for (_, m), i in zip(exps[method], mids)]
        ax.plot(lrs, ys, label=cfg['label'], color=cfg['color'],
                marker=cfg['marker'], linestyle=cfg['ls'], linewidth=1.8, markersize=7)
    if extra_diag:
        ax.plot([1e-5, 3e-2], [1e-5, 3e-2], 'k--', alpha=0.3, label='lr_eff = lr')
    ax.set_xscale('log')
    if use_ylog: ax.set_yscale('log')
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    _savefig(out_dir, fname)


def _plot_vs_step(exps, out_dir, fname, title, ylabel, metric_key,
                  smooth=False, xlim=None, ylim=None, use_ylog=False):
    """Generic 'metric vs step, one curve per method at that method's best LR' plot."""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    for method, cfg in SWEEPS.items():
        if method not in exps or not has_key(exps, method, metric_key):
            continue
        best_lr, best_m = min(exps[method], key=lambda x: final_loss(x[1]))
        steps = best_m['step']
        vals = best_m[metric_key]
        if smooth: vals = ema_smooth(vals, alpha=0.05)
        ax.plot(steps, vals, label=f"{cfg['label']} (lr={best_lr:.0e})",
                color=cfg['color'], linestyle=cfg['ls'], linewidth=1.5, alpha=0.9)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if use_ylog: ax.set_yscale('log')
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, loc='upper right', ncol=2 if smooth else 1)
    ax.grid(True, alpha=0.3, which='both' if use_ylog else 'major')
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    _savefig(out_dir, fname)


def plot_lr_sweep(exps, out_dir, title_suffix=''):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    for method, cfg in SWEEPS.items():
        if method not in exps:
            continue
        lrs = [lr for lr, _ in exps[method]]
        losses = [final_loss(m) for _, m in exps[method]]
        if method in HORIZONTAL_METHODS:
            mean_l, std_l = np.mean(losses), np.std(losses)
            ax.axhline(mean_l, color=cfg['color'], linestyle=cfg['ls'],
                       linewidth=1.8, label=cfg['label'])
            ax.axhspan(mean_l - std_l, mean_l + std_l, color=cfg['color'], alpha=0.15)
        else:
            ax.plot(lrs, losses, label=cfg['label'], color=cfg['color'],
                    marker=cfg['marker'], linestyle=cfg['ls'], linewidth=1.8, markersize=7)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Final Train Loss', fontsize=12)
    title = f'LR Sweep — Final Loss{title_suffix}'
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(2.5, 8.0)
    _savefig(out_dir, 'lr_sweep_loss.png')


def _infer_title_suffix(exps):
    """Infer ' (N steps, ~XM params, K ctx)' suffix from any loaded run."""
    for method in exps:
        if not exps[method]:
            continue
        _, m = exps[method][0]
        n_steps = m['step'][-1] if 'step' in m and m['step'] else None
        n_params = m.get('n_params', [None])[0] if isinstance(m.get('n_params'), list) else m.get('n_params')
        bits = []
        if n_steps:
            bits.append(f'{n_steps} steps')
        if n_params:
            bits.append(f'~{round(n_params/1e6)}M params')
        if bits:
            return ' (' + ', '.join(bits) + ')'
    return ''


def make_all_plots(exp_root):
    experiments_dir = os.path.join(exp_root, 'experiments')
    out_dir = os.path.join(exp_root, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    exps = load_all(experiments_dir, SWEEPS.keys())
    print(f'Loaded {sum(len(v) for v in exps.values())} jobs across {len(exps)} experiments\n')

    suffix = _infer_title_suffix(exps)

    plot_lr_sweep(exps, out_dir, title_suffix=suffix)
    _plot_vs_lr(exps, out_dir, 'dampening_vs_lr.png',
                'Adaptive Dampening vs LR' + suffix,
                'Dampening Factor (mid-training)', 'optim/dampening')
    _plot_vs_lr(exps, out_dir, 'lr_eff_vs_lr.png',
                'Effective LR vs Nominal LR' + suffix,
                'Effective Learning Rate (mid-training)', 'optim/lr_eff',
                extra_diag=True)
    # last training step (for xlim) — taken from any run
    last_step = None
    for method in exps:
        if exps[method]:
            last_step = exps[method][0][1]['step'][-1]
            break
    _plot_vs_step(exps, out_dir, 'training_curves_best.png',
                  'Training Curves — Best LR per Method (EMA smoothed)' + suffix,
                  'Train Loss', 'train/loss',
                  smooth=True, xlim=(0, last_step) if last_step else None,
                  ylim=(2.5, 8.0), use_ylog=True)
    _plot_vs_step(exps, out_dir, 'dual_norm_training.png',
                  'Dual Norm² over Training — Best LR per Method' + suffix,
                  'Dual Norm Squared', 'optim/dual_norm_sq')
    _plot_vs_step(exps, out_dir, 'grad_norm_training.png',
                  'Gradient Norm over Training — Best LR per Method' + suffix,
                  'Gradient Norm', 'optim/grad_norm')
    _plot_vs_step(exps, out_dir, 'lr_eff_training.png',
                  'Effective LR over Training — Best LR per Method' + suffix,
                  'Effective LR', 'optim/lr_eff')
    print(f'\nAll plots saved to {out_dir}/')


def main():
    exp_root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath('exps_70m')
    make_all_plots(exp_root)


if __name__ == '__main__':
    main()
