"""
Compare different "averages" for combining eta with the SCION Polyak step.

For each method in exps_70m/experiments, plots a 3x3 grid of panels where
each panel shows one candidate average over the training steps, one curve
per LR (runs whose final loss diverged are dropped).

Usage:
    python -m plotting.plot_averages [exp_root]     # default: exps_70m
"""
import os, sys, math
from ._common import plt, load_runs, final_loss, polyak_step


AVG_NAMES = [
    'min', 'harmonic', 'parallel', 'geometric',
    'quadratic', 'max',
    'weighted_a=0.25', 'weighted_a=0.5', 'weighted_a=0.75',
]

DIVERGENCE_THRESHOLD = 5.0


def averages(a_list, b_list):
    """Compute all 9 averages element-wise from (eta, polyak_step)."""
    out = {
        'min':       [min(a, b) for a, b in zip(a_list, b_list)],
        'harmonic':  [2 * a * b / (a + b) if (a + b) > 0 else 0 for a, b in zip(a_list, b_list)],
        'parallel':  [a * b / (a + b) if (a + b) > 0 else 0 for a, b in zip(a_list, b_list)],
        'geometric': [math.sqrt(max(a * b, 0)) for a, b in zip(a_list, b_list)],
        'quadratic': [math.sqrt((a * a + b * b) / 2.0) for a, b in zip(a_list, b_list)],
        'max':       [max(a, b) for a, b in zip(a_list, b_list)],
    }
    for alpha in (0.25, 0.5, 0.75):
        out[f'weighted_a={alpha}'] = [alpha * a + (1 - alpha) * b
                                       for a, b in zip(a_list, b_list)]
    return out


def plot_per_method(experiments_dir, out_dir, method):
    runs = [(lr, m) for lr, m in load_runs(experiments_dir, method)
            if 'optim/dual_norm_sq' in m and final_loss(m) < DIVERGENCE_THRESHOLD]
    if not runs:
        print(f'  no runs for {method}')
        return

    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(runs) - 1, 1)) for i in range(len(runs))]

    fig, axes = plt.subplots(3, 3, figsize=(12, 8.5), sharex=True, sharey=True)

    for ax, name in zip(axes.flat, AVG_NAMES):
        for (lr, m), color in zip(runs, colors):
            steps = m['step']
            a_list = [lr] * len(steps)
            b_list = polyak_step(m['train/loss'], m['optim/dual_norm_sq'])
            vals = averages(a_list, b_list)[name]
            ax.plot(steps, vals, color=color, lw=1.0, alpha=0.9,
                    label=f'$\\eta$={lr:.0e}')
        ax.set_yscale('log')
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(labelsize=8)

    for ax in axes[-1, :]: ax.set_xlabel('step', fontsize=9)
    for ax in axes[:, 0]:  ax.set_ylabel(r'$\eta_k^{\mathrm{eff}}$', fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=7,
               bbox_to_anchor=(1.0, 0.5), framealpha=0.9, borderpad=0.3,
               labelspacing=0.3, handlelength=1.2)
    fig.suptitle(f'{method}: averages of $\\eta$ and $2 f_k / \\|g_k\\|_*^2$',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    out = os.path.join(out_dir, f'averages_{method}.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'  saved {out}')


def main():
    exp_root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath('exps_70m')
    experiments_dir = os.path.join(exp_root, 'experiments')
    out_dir = os.path.join(exp_root, 'plots', 'averages')
    os.makedirs(out_dir, exist_ok=True)
    print(f'EXP_ROOT = {exp_root}')
    for method in ['standard', 'L1', 'L1_unc']:
        print(f'\n--- {method} ---')
        plot_per_method(experiments_dir, out_dir, method)
    print(f'\nAll average plots saved to {out_dir}/')


if __name__ == '__main__':
    main()
