"""Generic training-curves plot for 70M / 160M / 410M.

Usage:
    python plotting/plot_training_curves.py --scale 70m
    python plotting/plot_training_curves.py --scale 160m
    python plotting/plot_training_curves.py --scale 410m

Writes exps_{scale}/training_curves.{png,pdf}. Picks each method's best-LR run
(finished only) and plots val/loss vs step.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
import wandb

from _style import apply_style, PALETTE, LW

SCALES = {
    '70m': dict(
        projects=['steeldream/scion_70m_final', 'steeldream/scion_70m_simplify'],
        title='70M Chinchilla',
        ylim=(3.3, 5.0),
        max_step=5500,
        out_dir='exps_70m',
    ),
    '160m': dict(
        projects=['steeldream/scion_160m'],
        title='160M Chinchilla',
        ylim=(2.9, 5.0),
        max_step=12200,
        out_dir='exps_160m',
    ),
    '410m': dict(
        projects=['steeldream/scion_410m_full_chinchilla'],
        title='410M Chinchilla',
        ylim=(2.6, 5.0),
        max_step=15680,
        out_dir='exps_410m',
    ),
}

METHODS = [
    ('Scion',                         'scion'),
    ('L-NGN-Scion',                   'lngn_scion'),
    (r'G-NGN-Scion ($\alpha=7\!\times\!10^{-2}$)', 'gngn_alpha_7e-2'),
    ('AdamW',                         'adamw'),
    ('Muonmax-Momo',                  'muonmax_momo'),
    ('NGN-MDv1',                      'ngnmdv1'),
]


def lr_from_name(name):
    try:
        return float(name.split('_lr_')[-1])
    except Exception:
        return None


def find_best_run(prefix, projects):
    api = wandb.Api()
    best_run, best_val, best_lr = None, float('inf'), None
    for proj in projects:
        for r in api.runs(proj, per_page=200):
            n = r.name or ''
            if not n.startswith(prefix + '_lr_') or r.state != 'finished':
                continue
            lr = lr_from_name(n)
            if lr is None:
                continue
            v = r.summary.get('valid/loss')
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                continue
            if v < best_val:
                best_val, best_run, best_lr = v, r, lr
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

    apply_style()

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    print(f'{"method":<26} {"opt lr":>10} {"final val":>10} {"n eval":>8}')
    print('-' * 60)
    for label, prefix in METHODS:
        run, val, lr = find_best_run(prefix, cfg['projects'])
        if run is None:
            print(f'{label:<26} (no data)')
            continue
        steps, vals = fetch_history(run)
        if len(steps) == 0:
            print(f'{label:<26} (no history)')
            continue
        print(f'{label:<26} {lr:>10.0e} {val:>10.3f} {len(steps):>8}')
        color = PALETTE[prefix]
        ax.plot(steps, vals, color=color, lw=LW,
                label=fr'{label} ($\eta={lr:.0e}$, val$={val:.3f}$)')

    ax.set_xlabel('Training step')
    ax.set_ylabel('Validation loss')
    ax.set_ylim(*cfg['ylim'])
    ax.set_xlim(0, cfg['max_step'])
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_title(cfg['title'])
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(repo_root, cfg['out_dir'], 'training_curves.png')
    plt.savefig(out, dpi=150)
    plt.savefig(out.replace('.png', '.pdf'))
    print(f'\nSaved {out} (+ .pdf)')


if __name__ == '__main__':
    main()
