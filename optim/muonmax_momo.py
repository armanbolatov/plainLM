"""
MuonMax-Momo optimizer from arXiv:2510.09827.
Adapted from https://github.com/modichirag/GPT-opt/blob/michael/gptopt/optim/muonmax_momo.py

Muon (polar/NS orthogonalization) for 2D weight matrices, Adam for the rest.
Optional model-based truncation (Momo) for adaptive step size.
"""

import torch


@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class MuonMaxMomo(torch.optim.Optimizer):
    """
    MuonMax-Momo (arXiv:2510.09827).

    Args:
        muon_params: list of 2D parameters for Muon (spectral) updates.
        adam_params: list of other parameters for Adam updates.
        lr: base learning rate (applied to Adam; Muon uses lr * muon_lr_scale).
        muon_lr_scale: multiplier for Muon lr relative to Adam lr.
        wd: weight decay.
        momentum: EMA coefficient for Muon momentum buffer.
        ns_steps: Newton-Schulz iterations for polar decomposition.
        betas: (beta1, beta2) for Adam.
        eps: epsilon for Adam.
        truncate_loss: lower bound for loss truncation (None to disable, 0.0 for default).
        stale_nuc: reuse nuclear norm from previous step (faster).
    """

    def __init__(self, muon_params, adam_params, *, lr=1e-3, muon_lr_scale=10.0,
                 wd=0.1, momentum=0.95, ns_steps=5, betas=(0.95, 0.95), eps=1e-8,
                 truncate_loss=0.0, stale_nuc=True):

        assert isinstance(muon_params, list)
        assert isinstance(adam_params, list)

        self.muon_lr_scale = muon_lr_scale
        self.ns_steps = ns_steps
        self.diagnostics = {}

        defaults = dict(lr=lr, wd=wd, momentum=momentum, betas=betas, eps=eps)
        params = list(muon_params) + list(adam_params)
        super().__init__(params, defaults)

        for p in muon_params:
            self.state[p]["muon"] = True
        for p in adam_params:
            self.state[p]["muon"] = False

        # Model truncation (Momo)
        self.truncate_loss = truncate_loss
        self.use_truncation = truncate_loss is not None
        if self.use_truncation:
            assert momentum == betas[0], "Momo requires momentum == beta1"
            self.loss_model = None
        self.stale_nuc = stale_nuc

    def step(self, closure=None, loss=None):
        if self.use_truncation:
            assert (closure is not None) or (loss is not None)
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        wd = group["wd"]
        momentum = group["momentum"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]

        # Phase 1: update momentum, compute dual norms
        grad_sq = 0.0
        muon_dual_norm = 0.0
        adam_sq_dual_norm = 0.0
        current_loss_model = 0.0
        new_loss_model = 0.0

        for p in group["params"]:
            g = p.grad
            if g is None:
                continue

            grad_sq += g.float().pow(2).sum().item()
            state = self.state[p]

            if state["muon"]:
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = g.clone()
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g, alpha=1.0 - momentum)
            else:
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = g.clone()
                    state["sq_momentum_buffer"] = g.square()
                buf = state["momentum_buffer"]
                buf2 = state["sq_momentum_buffer"]
                buf.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

            # Model truncation terms
            if self.use_truncation:
                current_loss_model += torch.sum(torch.mul(p.data, p.grad.data))
                new_loss_model += torch.sum(torch.mul(p.data, buf.data))

            # Dual norm
            if state["muon"]:
                if not self.stale_nuc or "prev_nuc_norm" not in state:
                    m = state["momentum_buffer"]
                    u = zeropower_via_newtonschulz5(m.view(m.shape[0], -1), steps=self.ns_steps).view(m.shape)
                    muon_dual_norm += (m * u).sum()
                else:
                    muon_dual_norm += state["prev_nuc_norm"]
            else:
                m = state["momentum_buffer"]
                v = state["sq_momentum_buffer"]
                adam_sq_dual_norm += torch.sum(m ** 2 / (eps + v.sqrt()))

        global_dual_norm = torch.sqrt(self.muon_lr_scale * muon_dual_norm ** 2 + adam_sq_dual_norm)
        d_sq = global_dual_norm.item() ** 2

        # Phase 2: Momo truncation
        current_lr = lr
        dampening = 1.0
        if self.use_truncation and loss is not None:
            loss_model_update = loss if isinstance(loss, float) else loss.item()
            loss_model_update -= current_loss_model.item()
            if self.loss_model is None:
                self.loss_model = loss_model_update
            self.loss_model = momentum * self.loss_model + (1 - momentum) * loss_model_update
            if d_sq > 1e-12:
                truncated_lr = (self.loss_model - self.truncate_loss + new_loss_model.item()) / d_sq
                current_lr = min(truncated_lr, lr)
                dampening = lr / current_lr if current_lr > 1e-12 else 1.0

        # Phase 3: Update Muon params
        update_sq = 0.0
        muon_nuc = muon_dual_norm.item() if isinstance(muon_dual_norm, torch.Tensor) else muon_dual_norm

        for p in group["params"]:
            state = self.state[p]
            if not state["muon"] or p.grad is None:
                continue
            m = state["momentum_buffer"]
            u = zeropower_via_newtonschulz5(m.view(m.shape[0], -1), steps=self.ns_steps).view(m.shape)
            p.data.mul_(1 - lr * wd)
            alpha = current_lr * self.muon_lr_scale * muon_nuc
            p.data.add_(u, alpha=-alpha)
            update_sq += (alpha ** 2) * u.float().pow(2).sum().item()

            if self.stale_nuc:
                state["prev_nuc_norm"] = (m * u).sum()

        # Phase 4: Update Adam params
        for p in group["params"]:
            state = self.state[p]
            if state["muon"] or p.grad is None:
                continue
            m = state["momentum_buffer"]
            v = state["sq_momentum_buffer"]
            direction = m / (eps + v.sqrt())
            p.data.mul_(1 - lr * wd)
            p.data.add_(direction, alpha=-current_lr)
            update_sq += (current_lr ** 2) * direction.float().pow(2).sum().item()

        self.diagnostics = {
            'optim/lr_eff': current_lr,
            'optim/dampening': dampening,
            'optim/grad_norm': grad_sq ** 0.5,
            'optim/update_norm': update_sq ** 0.5,
            'optim/dual_norm_sq': d_sq,
            'optim/dn_layer_std': 0.0,
            'optim/dn_layer_max': muon_nuc,
        }
        return loss
