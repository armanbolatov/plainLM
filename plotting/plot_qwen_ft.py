"""SFT optimizer comparison across Qwen models: val loss vs learning rate.

Four panels -- Qwen2.5-{0.5B, 1.5B, 3B} and Qwen3-4B -- from the local
metrics.json written by qwen_ft/finetune.py. NGN-Scion runs with the single
alpha=2e-2 tuned once on 0.5B; the Qwen3-4B panel adds the alpha=5e-2
diagnostic showing the alpha optimum is family-dependent.

    python plotting/plot_qwen_ft.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt

from _style import apply_style, minor_grid, PALETTE, LW, S_LINE

# (legend label, palette key, marker) shared across panels
GNGN = ('NGN-Scion', 'gngn_alpha_7e-2', 'v')
ADAMW = ('AdamW', 'adamw', 'P')
MUON = ('Muonmax-Momo', 'muonmax_momo', 'X')
NGNMD = ('NGN-MDv1', 'ngnmdv1', 'h')
SFP = ('ScheduleFree+', 'sfplus', 'D')
SCION = ('Scion', 'scion', 'o')

PANELS = [
    ('Qwen2.5-0.5B', [('gngn_p2',) + GNGN, ('adamw',) + ADAMW,
                      ('muonmax',) + MUON, ('scion_tuned',) + SCION,
                      ('ngnmdv1',) + NGNMD, ('sfplus',) + SFP]),
    ('Qwen2.5-1.5B', [('15b_gngn_p2',) + GNGN, ('15b_adamw',) + ADAMW,
                      ('15b_muonmax',) + MUON, ('15b_scion_tuned',) + SCION,
                      ('15b_ngnmdv1',) + NGNMD, ('15b_sfplus',) + SFP]),
    ('Qwen2.5-3B',   [('3b_gngn_p2',) + GNGN, ('3b_adamw',) + ADAMW,
                      ('3b_muonmax',) + MUON, ('3b_scion_tuned',) + SCION,
                      ('3b_ngnmdv1',) + NGNMD, ('3b_sfplus',) + SFP]),
]

RESULTS = 'qwen_ft/results_const'
LR_GRID = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
YLIM = (1.05, 5.0)


def load(tag):
    out = {}
    for f in glob.glob(os.path.join(RESULTS, tag, 'lr_*', 'metrics.json')):
        d = json.load(open(f))
        v = d.get('final_val_loss')
        on_grid = any(abs(d['lr'] - g) / g < 0.01 for g in LR_GRID) or d['lr'] == 1.0
        if isinstance(v, (int, float)) and v == v and v <= YLIM[1] and on_grid:
            out[d['lr']] = v
    return out


def main():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True,
                             sharey=True)
    for ax, (title, series) in zip(axes, PANELS):
        for tag, label, key, marker in series:
            d = load(tag)
            if not d:
                continue
            if tag.endswith('psf') or tag.endswith('sfplus'):
                # LR-free method: its lr argument is not a real learning rate,
                # so draw the single result as a horizontal reference line.
                ax.axhline(list(d.values())[0], color=PALETTE[key],
                           ls=':', lw=LW, label=label)
                continue
            lrs = sorted(d)
            ax.plot(lrs, [d[l] for l in lrs], color=PALETTE[key], lw=LW,
                    label=label)
            ax.scatter(lrs, [d[l] for l in lrs], color=PALETTE[key],
                       marker=marker, s=S_LINE, zorder=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(*YLIM)
        ax.set_xlabel('Learning rate')
        minor_grid(ax, axis='y')
        ax.text(0.5, 0.94, title, transform=ax.transAxes,
                ha='center', va='top', fontsize=12, color='#2A2A2A')

    axes[0].set_ylabel('Validation loss')
    # One legend from the union of panels (later panels add the alpha=5e-2 entry).
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    # One row, as in the pretraining robustness figure.
    fig.legend(handles, labels, loc='outside lower center',
               ncols=len(handles), frameon=False, fontsize=9)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'paper', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'qwen_ft.png')
    plt.savefig(out, dpi=150)
    plt.savefig(out.replace('.png', '.pdf'))
    print(f'Saved {out} (+ .pdf)')


if __name__ == '__main__':
    main()
