"""Generate sweep comparison plots for SCION variants + baselines."""
import json, glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EXPERIMENTS_DIR = '/home/arman/plainLM/experiments'
OUTPUT_DIR = '/home/arman/plainLM/plots'

SWEEPS = {
    'L1':         {'label': 'SCION NGN (constr)',       'color': '#9467bd', 'marker': 'D',  'ls': '--'},
    'L1_unc':     {'label': 'SCION NGN (unconstr)',    'color': '#8c564b', 'marker': 'P',  'ls': '--'},
    'standard':   {'label': 'Standard SCION',          'color': '#000000', 'marker': 'x',  'ls': ':'},
    'adamw':      {'label': 'AdamW',                   'color': '#e377c2', 'marker': '*',  'ls': '-'},
    'ngnmdv1':    {'label': 'NGN-MDv1',                'color': '#17becf', 'marker': 'h',  'ls': '-'},
    'muonmax_momo': {'label': 'MuonMax-Momo',          'color': '#bcbd22', 'marker': 'd',  'ls': '-'},
}

def load_all():
    exps = {}
    # Support both old (job_idx_N) and new (lr_X) directory structures
    for pattern in [f'{EXPERIMENTS_DIR}/*/lr_*/metrics.json',
                    f'{EXPERIMENTS_DIR}/*/job_idx_*/metrics.json']:
        for path in sorted(glob.glob(pattern)):
            parts = path.split('/')
            exp = parts[-3]
            if exp not in SWEEPS:
                continue
            with open(path) as f:
                m = json.load(f)
            peak_lr = max(m['lr']) if isinstance(m['lr'], list) else m['lr']
            if exp not in exps:
                exps[exp] = []
            # avoid duplicates
            if not any(abs(lr - peak_lr) < 1e-10 for lr, _ in exps[exp]):
                exps[exp].append((peak_lr, m))
    for exp in exps:
        exps[exp].sort(key=lambda x: x[0])
    return exps

def has_key(exps, exp, key):
    if exp not in exps or not exps[exp]:
        return False
    return key in exps[exp][0][1]

def final_loss(m):
    """Get final loss: prefer train/loss_avg if available, else last train/loss."""
    if 'train/loss_avg' in m and m['train/loss_avg']:
        return m['train/loss_avg'][-1]
    return m['train/loss'][-1]

# ── LR sweep ────────────────────────────────────────────────────────────────
def plot_lr_sweep(exps):
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp, cfg in SWEEPS.items():
        if exp not in exps:
            continue
        lrs = [lr for lr, m in exps[exp]]
        losses = [final_loss(m) for lr, m in exps[exp]]
        ax.plot(lrs, losses, label=cfg['label'], color=cfg['color'],
                marker=cfg['marker'], linestyle=cfg['ls'], linewidth=1.8, markersize=7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Final Train Loss', fontsize=12)
    ax.set_title('LR Sweep — Final Loss (5500 steps, 72M, 1024 ctx)', fontsize=13)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(3.2, 8.0)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lr_sweep_loss.png', dpi=150)
    plt.close()
    print('saved lr_sweep_loss.png')

# ── Dampening vs LR ─────────────────────────────────────────────────────────
def plot_dampening_vs_lr(exps):
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp, cfg in SWEEPS.items():
        if exp not in exps or not has_key(exps, exp, 'optim/dampening'):
            continue
        lrs = [lr for lr, m in exps[exp]]
        mid_idx = [len(m['step']) // 2 for lr, m in exps[exp]]
        damps = [m['optim/dampening'][i] for (lr, m), i in zip(exps[exp], mid_idx)]
        ax.plot(lrs, damps, label=cfg['label'], color=cfg['color'],
                marker=cfg['marker'], linestyle=cfg['ls'], linewidth=1.8, markersize=7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Dampening Factor (mid-training)', fontsize=12)
    ax.set_title('Adaptive Dampening vs LR', fontsize=13)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/dampening_vs_lr.png', dpi=150)
    plt.close()
    print('saved dampening_vs_lr.png')

# ── Effective LR vs nominal LR ──────────────────────────────────────────────
def plot_lr_eff_vs_lr(exps):
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp, cfg in SWEEPS.items():
        if exp not in exps or not has_key(exps, exp, 'optim/lr_eff'):
            continue
        lrs = [lr for lr, m in exps[exp]]
        mid_idx = [len(m['step']) // 2 for lr, m in exps[exp]]
        lr_effs = [m['optim/lr_eff'][i] for (lr, m), i in zip(exps[exp], mid_idx)]
        ax.plot(lrs, lr_effs, label=cfg['label'], color=cfg['color'],
                marker=cfg['marker'], linestyle=cfg['ls'], linewidth=1.8, markersize=7)
    lrs_range = [1e-5, 3e-2]
    ax.plot(lrs_range, lrs_range, 'k--', alpha=0.3, label='lr_eff = lr')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Nominal Learning Rate', fontsize=12)
    ax.set_ylabel('Effective Learning Rate (mid-training)', fontsize=12)
    ax.set_title('Effective LR vs Nominal LR', fontsize=13)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lr_eff_vs_lr.png', dpi=150)
    plt.close()
    print('saved lr_eff_vs_lr.png')

# ── Training curves — best LR per method ────────────────────────────────────
def plot_training_curves_best(exps):
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp, cfg in SWEEPS.items():
        if exp not in exps:
            continue
        best_lr, best_m = min(exps[exp], key=lambda x: final_loss(x[1]))
        steps = best_m['step']
        losses = best_m['train/loss']
        ax.plot(steps, losses, label=f"{cfg['label']} (lr={best_lr:.0e})",
                color=cfg['color'], linestyle=cfg['ls'], linewidth=1.5, alpha=0.85)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Train Loss', fontsize=12)
    ax.set_title('Training Curves — Best LR per Method', fontsize=13)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(3.2, 7.0)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/training_curves_best.png', dpi=150)
    plt.close()
    print('saved training_curves_best.png')

# ── Dual norm² over training ────────────────────────────────────────────────
def plot_dual_norm_over_training(exps):
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp, cfg in SWEEPS.items():
        if exp not in exps or not has_key(exps, exp, 'optim/dual_norm_sq'):
            continue
        best_lr, best_m = min(exps[exp], key=lambda x: final_loss(x[1]))
        steps = best_m['step']
        dn_sq = best_m['optim/dual_norm_sq']
        ax.plot(steps, dn_sq, label=f"{cfg['label']} (lr={best_lr:.0e})",
                color=cfg['color'], linestyle=cfg['ls'], linewidth=1.5, alpha=0.85)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Dual Norm Squared', fontsize=12)
    ax.set_title('Dual Norm² over Training — Best LR per Method', fontsize=13)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/dual_norm_training.png', dpi=150)
    plt.close()
    print('saved dual_norm_training.png')

# ── Gradient norm over training ──────────────────────────────────────────────
def plot_grad_norm_over_training(exps):
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp, cfg in SWEEPS.items():
        if exp not in exps:
            continue
        if not has_key(exps, exp, 'optim/grad_norm'):
            continue
        best_lr, best_m = min(exps[exp], key=lambda x: final_loss(x[1]))
        steps = best_m['step']
        gn = best_m['optim/grad_norm']
        ax.plot(steps, gn, label=f"{cfg['label']} (lr={best_lr:.0e})",
                color=cfg['color'], linestyle=cfg['ls'], linewidth=1.5, alpha=0.85)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Norm over Training — Best LR per Method', fontsize=13)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/grad_norm_training.png', dpi=150)
    plt.close()
    print('saved grad_norm_training.png')

# ── Effective LR over training ───────────────────────────────────────────────
def plot_lr_eff_over_training(exps):
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp, cfg in SWEEPS.items():
        if exp not in exps or not has_key(exps, exp, 'optim/lr_eff'):
            continue
        best_lr, best_m = min(exps[exp], key=lambda x: final_loss(x[1]))
        steps = best_m['step']
        lr_eff = best_m['optim/lr_eff']
        ax.plot(steps, lr_eff, label=f"{cfg['label']} (lr={best_lr:.0e})",
                color=cfg['color'], linestyle=cfg['ls'], linewidth=1.5, alpha=0.85)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Effective LR', fontsize=12)
    ax.set_title('Effective LR over Training — Best LR per Method', fontsize=13)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lr_eff_training.png', dpi=150)
    plt.close()
    print('saved lr_eff_training.png')

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    exps = load_all()
    print(f'Loaded {sum(len(v) for v in exps.values())} jobs across {len(exps)} experiments\n')
    plot_lr_sweep(exps)
    plot_dampening_vs_lr(exps)
    plot_lr_eff_vs_lr(exps)
    plot_training_curves_best(exps)
    plot_dual_norm_over_training(exps)
    plot_grad_norm_over_training(exps)
    plot_lr_eff_over_training(exps)
    print(f'\nAll plots saved to {OUTPUT_DIR}/')
