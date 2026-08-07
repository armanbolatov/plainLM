"""Proximal-model view: next-iterate loss and stability index vs step size.

An analog of Figure 1 in Schaipp/Gower/Taylor, extended to spectral geometry.
Five algorithms are compared on a 2x2-matrix version of the paper's test
function

    f(X) = log(1 + exp(-Tr(AX))) + max{Tr(AX) - 2, 0}

with A a fixed non-symmetric 2x2 matrix. The loss still depends only on the
scalar Tr(AX), but using a matrix argument makes the spectral-norm proximal
step well-defined. The initial point satisfies Tr(A X_0) = -3, matching the
paper's x_t = -3.

They cover the (surrogate model) x (distance norm) grid, with SPS kept as a
Euclidean-only reference:

    model / norm    Frobenius       spectral
    linear          SGD             Scion
    NGN sqrt-lin.   NGN             NGN-Scion
    SPS truncated   SPS             --

Within either norm the linear model's stability index grows without bound
while both capped models saturate, so saturation is a property of the model
and carries over to any distance norm. The saturating methods are exactly the
ones that are learning-rate robust in the LM experiments.

The three Frobenius methods are cross-checked against the closed-form delta
in the paper.

Usage:
    python plotting/plot_proximal_stepsize.py

Writes paper/figures/proximal_stepsize.{png,pdf}.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np

from _style import apply_style, minor_grid, shared_legend, LW

# ----------------------------------------------------------------------------
# Problem.
# ----------------------------------------------------------------------------

A = np.array([[1.0,  0.5],
              [-0.3, 0.8]])          # fixed non-symmetric 2x2

# Want Tr(A X_0) = -3. d Tr(AX)/dX = A^T, so X = c * A^T makes
# Tr(A X) = c * Tr(A A^T) = c * ||A||^2_F.
TR_TARGET = -3.0
X0 = (TR_TARGET / np.sum(A * A)) * A.T
assert abs(np.trace(A @ X0) - TR_TARGET) < 1e-10


def f(X):
    t = np.trace(A @ X)
    return float(np.log1p(np.exp(-t)) + max(t - 2.0, 0.0))


def grad(X):
    t = float(np.trace(A @ X))
    # d/dt of log(1+exp(-t)) = -1/(1+exp(t)).
    # d/dt of max(t-2, 0)    = 1 if t>2 else 0 (the kink is ignored).
    df_dt = -1.0 / (1.0 + np.exp(t)) + (1.0 if t > 2.0 else 0.0)
    return df_dt * A.T              # d Tr(AX) / dX = A^T


def lmo_spec(G):
    """LMO of the unit spectral-norm ball: argmin_{||V||_spec<=1} <G,V> = -U V^T."""
    U, _, Vt = np.linalg.svd(G, full_matrices=False)
    return -U @ Vt                  # step Y = X + a*lmo_spec(g) points along -g


def frob_norm(M): return float(np.linalg.norm(M))
def spec_norm(M): return float(np.linalg.svd(M, compute_uv=False).max())
def nuc_norm(M):  return float(np.linalg.svd(M, compute_uv=False).sum())


# ----------------------------------------------------------------------------
# Methods. Each returns (X_next, model_value_at_X_next, dist^2 in method norm).
# ----------------------------------------------------------------------------

def sgd(X, alpha):
    g = grad(X)
    Y = X - alpha * g
    model = f(X) + float(np.sum(g * (Y - X)))         # f(x) + <g, y-x>
    return Y, model, frob_norm(Y - X) ** 2


def ngn(X, alpha):
    g = grad(X); fx = f(X); gn2 = frob_norm(g) ** 2
    gamma = alpha / (1.0 + (alpha / (2.0 * max(fx, 1e-12))) * gn2)
    Y = X - gamma * g
    sf = np.sqrt(max(fx, 1e-12))
    inner = float(np.sum(g * (Y - X)))
    model = (sf + inner / (2.0 * sf)) ** 2            # (sqrt f + <g,y-x>/2 sqrt f)^2
    return Y, model, frob_norm(Y - X) ** 2


def sps(X, alpha, C=0.0):
    g = grad(X); fx = f(X); gn2 = frob_norm(g) ** 2
    tau = min(alpha, (fx - C) / max(gn2, 1e-12))
    Y = X - tau * g
    model = max(f(X) + float(np.sum(g * (Y - X))), C)
    return Y, model, frob_norm(Y - X) ** 2


def spectral_gd(X, alpha):
    g = grad(X)
    Y = X + alpha * lmo_spec(g)
    model = f(X) + float(np.sum(g * (Y - X)))         # linear model, as in SGD
    return Y, model, spec_norm(Y - X) ** 2            # spectral norm


def ngn_spectral_gd(X, alpha):
    g = grad(X); fx = f(X)
    gn_nuc2 = nuc_norm(g) ** 2
    gamma = alpha / (1.0 + (alpha / (2.0 * max(fx, 1e-12))) * gn_nuc2)
    Y = X + gamma * lmo_spec(g)
    sf = np.sqrt(max(fx, 1e-12))
    inner = float(np.sum(g * (Y - X)))
    model = (sf + inner / (2.0 * sf)) ** 2
    return Y, model, spec_norm(Y - X) ** 2


# The (surrogate model) x (distance norm) grid, encoded so the structure is
# readable off the legend: colour = model, line style = norm.
#   solid  = Frobenius distance      dashed = spectral distance
# Listed model-major, so with ncols=3 each legend column is one model.
LINEAR, NGN_M, SPS_M = '#0173b2', '#de8f05', '#029e73'

METHODS = [
    ('SGD',           sgd,             LINEAR, '-'),
    ('Scion',         spectral_gd,     LINEAR, '--'),
    ('NGN',           ngn,             NGN_M,  '-'),
    ('NGN-Scion',   ngn_spectral_gd, NGN_M,  '--'),
    ('SPS',           sps,             SPS_M,  '-'),
]


def main():
    alphas = np.linspace(0.05, 40, 800)    # far enough for both NGN variants to saturate

    results = {}
    for name, step_fn, color, ls in METHODS:
        next_loss, deltas = [], []
        for a in alphas:
            Yn, model_val, dist2 = step_fn(X0, a)
            next_loss.append(f(Yn))
            # delta_t = f(x_t) - f_x(x_{t+1}) - ||x_{t+1} - x_t||^2 / (2 alpha)
            deltas.append(f(X0) - model_val - dist2 / (2.0 * a))
        results[name] = (np.asarray(next_loss), np.asarray(deltas), color, ls)

    # Cross-check the three Euclidean methods against the paper's closed forms.
    g0 = grad(X0); fx0 = f(X0); gn02 = frob_norm(g0) ** 2
    print('Cross-check against paper closed forms (max abs diff over alpha grid):')
    for name, closed in [
        ('SGD', lambda a: 0.5 * a * gn02),
        ('NGN', lambda a: 0.5 * (a / (1 + a / (2 * fx0) * gn02)) * gn02),
        ('SPS', lambda a: min(a, fx0 / gn02) * (1 - min(a, fx0 / gn02) / (2 * a)) * gn02),
    ]:
        num = results[name][1]
        cf = np.asarray([closed(a) for a in alphas])
        print(f'  {name:5s}  max |delta_numerical - delta_closedform| = '
              f'{np.abs(num - cf).max():.3e}')

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    for name, (loss, delta, color, ls) in results.items():
        axes[0].plot(alphas, loss,  color=color, linestyle=ls, lw=LW, label=name)
        axes[1].plot(alphas, delta, color=color, linestyle=ls, lw=LW, label=name)

    # Saturation levels (alpha -> infinity) for the cap-bounded methods.
    colors = {n: c for n, _, c, _ in METHODS}
    sat = {'SPS': fx0}
    sat['NGN'] = f(X0 - (2 * fx0 / frob_norm(g0) ** 2) * g0)
    sat['NGN-Scion'] = f(X0 + (2 * fx0 / nuc_norm(g0) ** 2) * lmo_spec(g0))
    for name, val in sat.items():
        axes[0].axhline(val, color=colors[name], linestyle=':', alpha=0.5, lw=1.2)

    axes[0].set_xlabel(r'Step size $\alpha$')
    axes[0].set_ylabel(r'$f(x_{t+1})$')
    axes[0].set_ylim(0, max(3.5, f(X0) * 1.1))
    axes[0].text(0.5, 0.94, 'Next-iterate loss', transform=axes[0].transAxes,
                 ha='center', va='top', fontsize=12, color='#2A2A2A')

    axes[1].set_xlabel(r'Step size $\alpha$')
    axes[1].set_ylabel(r'Stability index $\delta_t$')
    axes[1].set_ylim(0, 10)
    axes[1].text(0.5, 0.94, 'Stability index', transform=axes[1].transAxes,
                 ha='center', va='top', fontsize=12, color='#2A2A2A')

    for ax in axes:
        minor_grid(ax)
    shared_legend(fig, axes[0], ncols=3)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, 'paper', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'proximal_stepsize.png')
    plt.savefig(out, dpi=150)
    plt.savefig(out.replace('.png', '.pdf'))
    print(f'\nSaved {out} (+ .pdf)')


if __name__ == '__main__':
    main()
