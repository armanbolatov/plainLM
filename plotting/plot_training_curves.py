"""Generic training-curves plot for 70M / 160M / 410M.

Usage:
    python plotting/plot_training_curves.py --scale 70m
    python plotting/plot_training_curves.py --scale 160m
    python plotting/plot_training_curves.py --scale 410m

Writes exps_{scale}/training_curves.png. For each method, picks the run at that
method's best LR (finished only) and plots its val/loss vs training step.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import wandb

SCALES = {
    '70m': dict(
        projects=['steeldream/scion_70m_final', 'steeldream/scion_70m_simplify'],
        title='70M Chinchilla training curves at each method\'s optimum LR (5500 steps)',
        ylim=(3.0, 8.0),
        out_dir='exps_70m',
    ),
    '160m': dict(
        projects=['steeldream/scion_160m'],
        title='160M Chinchilla training curves at each method\'s optimum LR (12200 steps)',
        ylim=(3.0, 8.0),
        out_dir='exps_160m',
    ),
    '410m': dict(
        projects=['steeldream/scion_410m_full_chinchilla'],
        title='410M Full Chinchilla training curves at each method\'s optimum LR (15680 steps)',
        ylim=(2.6, 8.0),
        out_dir='exps_410m',
    ),
}

METHODS = [
    ('Scion',                   'scion',           'C0', '-'),
    ('L-NGN-Scion',             'lngn_scion',      'C1', '-'),
    ('G-NGN-Scion (α=7e-2)',    'gngn_alpha_7e-2', 'C2', '-'),
    ('AdamW (β=.9/.95)',        'adamw',           'C3', '-'),
    ('muonmax_momo',            'muonmax_momo',    'C4', '-'),
    ('NGN-MDv1',                'ngnmdv1',         'C5', '-'),
]


def lr_from_name(name):
    try:
        return float(name.split('_lr_')[-1])
    except Exception:
        return None


def find_best_run(prefix, projects):
    api = wandb.Api()
    best_run = None
    best_val = float('inf')
    best_lr = None
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
                continue
            if v < best_val:
                best_val = v
                best_run = r
                best_lr = lr
    return best_run, best_val, best_lr


def fetch_history(run, key='valid/loss', step_key='step'):
    steps, vals = [], []
    for row in run.scan_history(keys=[step_key, key], page_size=2000):
        s = row.get(step_key)
        v = row.get(key)
        if s is None or v is None or not np.isfinite(v):
            continue
        steps.append(s)
        vals.append(v)
    return np.array(steps), np.array(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scale', required=True, choices=list(SCALES.keys()))
    args = p.parse_args()
    cfg = SCALES[args.scale]

    fig, ax = plt.subplots(figsize=(10, 6))
    print(f'{"method":<24} {"opt lr":>10} {"final val":>10} {"n eval":>8}')
    print('-' * 60)
    for label, prefix, color, ls in METHODS:
        run, val, lr = find_best_run(prefix, cfg['projects'])
        if run is None:
            print(f'{label:<24} (no data)')
            continue
        steps, vals = fetch_history(run)
        if len(steps) == 0:
            print(f'{label:<24} (no history)')
            continue
        print(f'{label:<24} {lr:>10.0e} {val:>10.3f} {len(steps):>8}')
        ax.plot(steps, vals, color=color, linestyle=ls, linewidth=2,
                label=f'{label} (lr={lr:.0e}, val={val:.3f})')

    ax.set_xlabel('training step')
    ax.set_ylabel('val/loss')
    ax.set_yscale('log')
    ax.set_ylim(*cfg['ylim'])
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title(cfg['title'])
    plt.tight_layout()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(repo_root, cfg['out_dir'], 'training_curves.png')
    plt.savefig(out, dpi=110, bbox_inches='tight')
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
