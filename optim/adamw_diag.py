"""Thin wrapper to add diagnostics to any optimizer."""

import torch


def wrap_with_diagnostics(optimizer):
    """Add self.diagnostics to an optimizer by wrapping its step method."""
    original_step = optimizer.step

    def step_with_diag(*args, **kwargs):
        # Compute grad norm before step
        grad_sq = 0.0
        param_snapshots = {}
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_sq += p.grad.float().pow(2).sum().item()
                    param_snapshots[id(p)] = p.data.clone()

        result = original_step(*args, **kwargs)

        # Compute update norm after step
        update_sq = 0.0
        for group in optimizer.param_groups:
            for p in group['params']:
                if id(p) in param_snapshots:
                    update_sq += (p.data - param_snapshots[id(p)]).float().pow(2).sum().item()

        optimizer.diagnostics = {
            'optim/grad_norm': grad_sq ** 0.5,
            'optim/update_norm': update_sq ** 0.5,
            'optim/lr_eff': optimizer.param_groups[0]['lr'],
            'optim/dampening': 1.0,
            'optim/dual_norm_sq': 0.0,
            'optim/dn_layer_std': 0.0,
            'optim/dn_layer_max': 0.0,
        }
        return result

    optimizer.step = step_with_diag
    optimizer.diagnostics = {}
    return optimizer
