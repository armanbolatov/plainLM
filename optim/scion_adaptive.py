"""SCION optimizer (Spectral Conditioning by Iterative Normalisation).

Five modes, selected by the `adaptive` argument:

    adaptive=False     Standard SCION (no adaptive step):
                         x ← (1 - lr) x - lr · scale · lmo(buf_g)

    adaptive='ngn'     SCION-NGN, harmonic mean of cfg.lr with the GLOBAL Polyak:
                         γ = lr · polyak_global / (lr + polyak_global)
                         x ← (1 - γ) x - γ · scale · lmo(buf_g)

    adaptive='sps'     Stochastic Polyak Stepsize, GLOBAL Polyak capped by cfg.lr:
                         γ = min(lr, polyak_global)
                         x ← (1 - γ) x - γ · scale · lmo(buf_g)

    adaptive='ngn_pl'  NGN with per-LAYER Polyak: each parameter l gets its own
                         γ_l = lr · polyak_l / (lr + polyak_l)

    adaptive='sps_pl'  SPS with per-LAYER Polyak:
                         γ_l = min(lr, polyak_l)

Global Polyak (uses Σ_l dn_l, then squares):
        polyak_global = 2 · loss / (Σ_l dn_l)²       where dn_l = ⟨g_l, lmo_l(g_l)⟩

Per-layer Polyak (each layer treated independently):
        polyak_l      = 2 · loss / dn_l²

The dual norm is computed on the raw gradient; momentum is only used inside the
update direction.

`polyak_multiplier` scales the Polyak step before combining with cfg.lr.
`unconstrained=True` drops the `(1 - γ) x` shrinkage so the step becomes plain
`x ← x - γ · scale · lmo(buf_g)`.
"""

import torch


# ----------------------------------------------------------------------------
# Newton–Schulz: zeroth power (orthogonalisation) of a 2D matrix in bf16.
# ----------------------------------------------------------------------------

@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    """Quintic Newton-Schulz iteration that approximates the orthogonal factor
    of G's SVD. Coefficients pick a steep slope at 0; the output has spectral
    norm at most 1 (modulo NS approximation noise)."""
    assert G.dim() == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)                       # spectral norm ≤ 1
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


# ----------------------------------------------------------------------------
# Norms / linear minimisation oracles (LMOs).
# ----------------------------------------------------------------------------

class Norm:
    def lmo(self, g):
        raise NotImplementedError


class Spectral(Norm):
    """LMO of the spectral norm ball. Returns Newton-Schulz orthogonalisation
    of g, optionally scaled by sqrt(d_out/d_in) or sqrt(d_out)."""

    def __init__(self, normalized=True, steps=5):
        self.normalized = normalized
        self.steps = steps

    def lmo(self, g):
        if g.dim() < 2:                              # fall back to Sign for 1-D tensors
            fan_in = max(g.numel() // g.shape[0], 1) if g.dim() >= 1 else 1
            return (1.0 / fan_in) * torch.sign(g)
        out = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        d_out, d_in = out.shape
        scale = (d_out / d_in) ** 0.5 if self.normalized else d_out ** 0.5
        return out * scale


class Sign(Norm):
    """LMO of the sign / l1 ball. Returns sign(g), optionally normalised by
    fan_in so the LMO has bounded element magnitude."""

    def __init__(self, normalized=True):
        self.normalized = normalized

    def lmo(self, g):
        if self.normalized:
            fan_in = max(g.numel() // g.shape[0], 1) if g.dim() >= 1 else 1
            return (1.0 / fan_in) * torch.sign(g)
        return torch.sign(g)


norm_dict = {'Spectral': Spectral, 'Sign': Sign}


# ----------------------------------------------------------------------------
# Optimizer.
# ----------------------------------------------------------------------------

_ADAPTIVE_VALID = (False, 'ngn', 'sps', 'ngn_pl', 'sps_pl')


class Scion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=1.0, norm='Spectral',
                 norm_kwargs=None, scale=1.0, unconstrained=False, weight_decay=0.0,
                 adaptive=False, polyak_multiplier=1.0, polyak_use_scale=False,
                 polyak_form=None):
        if lr < 0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0:
            raise ValueError(f'Invalid momentum: {momentum}')
        if adaptive not in _ADAPTIVE_VALID:
            raise ValueError(f"adaptive must be one of {_ADAPTIVE_VALID}, got {adaptive!r}")
        if norm_kwargs is None:
            norm_kwargs = {}

        defaults = dict(lr=lr, momentum=momentum, norm=norm, norm_kwargs=norm_kwargs,
                        scale=scale, unconstrained=unconstrained, weight_decay=weight_decay,
                        polyak_multiplier=polyak_multiplier)
        self.adaptive = adaptive
        self.polyak_multiplier = polyak_multiplier
        # Backwards compatibility: polyak_use_scale=True maps to 'sqr_sdn'.
        if polyak_form is None:
            polyak_form = 'sqr_sdn' if polyak_use_scale else 'sqr_dn'
        if polyak_form not in ('sqr_dn', 'sqr_sdn', 'lin_sdn'):
            raise ValueError(f"polyak_form must be 'sqr_dn', 'sqr_sdn', or 'lin_sdn', got {polyak_form!r}")
        self.polyak_form = polyak_form
        self.diagnostics = {}
        super().__init__(params, defaults)

    # ------------------------------------------------------------------------
    # Public dispatch.
    # ------------------------------------------------------------------------

    def step(self, loss=None):
        if self.adaptive:
            self._step_adaptive(loss)
        else:
            self._step_standard()

    # ------------------------------------------------------------------------
    # Standard SCION (no adaptive step).
    # ------------------------------------------------------------------------

    def _step_standard(self):
        grad_sq, update_sq = 0.0, 0.0
        dn_list = []
        lr = 0.0

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            scale = group['scale']
            unconstrained = group['unconstrained']
            weight_decay = group.get('weight_decay', 0.0)
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])

            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                grad_sq += g.float().pow(2).sum().item()
                dn_list.append((g * norm_backend.lmo(g)).sum().item())

                # momentum buffer on raw gradient
                if momentum != 1:
                    buf = state.setdefault('momentum_buffer', torch.zeros_like(g))
                    buf.mul_(1 - momentum).add_(g, alpha=momentum)
                    g = buf

                lmo_g = norm_backend.lmo(g)
                update_sq += (lr * scale) ** 2 * lmo_g.float().pow(2).sum().item()

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                if unconstrained:
                    p.data.add_(lmo_g, alpha=-lr * scale)
                else:
                    p.data.mul_(1 - lr).add_(lmo_g, alpha=-lr * scale)

        self._log_diagnostics(lr, dn_list, grad_sq, update_sq, dampening=1.0,
                              dual_norm_sq=None, polyak=None)

    # ------------------------------------------------------------------------
    # NGN / SPS (adaptive step using the SCION global dual norm).
    # ------------------------------------------------------------------------

    def _step_adaptive(self, loss):
        if loss is None:
            raise ValueError(f"adaptive={self.adaptive!r} requires loss. Pass loss=val to step().")

        per_layer = self.adaptive.endswith('_pl')
        mode = self.adaptive[:-3] if per_layer else self.adaptive  # 'ngn' or 'sps'

        # Pass 1: per-layer dual norm on RAW gradient, then update momentum buffers
        # and cache LMOs of the buffered gradient for the update pass.
        dn_list, grad_sq = [], 0.0
        cached = []
        for group in self.param_groups:
            momentum = group['momentum']
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            group_cache = []
            for p in group['params']:
                g = p.grad
                if g is None:
                    group_cache.append((p, None, 0.0))
                    continue
                state = self.state[p]

                grad_sq += g.float().pow(2).sum().item()
                dn = (g * norm_backend.lmo(g)).sum().item()
                dn_list.append(dn)

                if momentum != 1:
                    buf = state.setdefault('momentum_buffer', torch.zeros_like(g))
                    buf.mul_(1 - momentum).add_(g, alpha=momentum)
                    g = buf

                group_cache.append((p, norm_backend.lmo(g), dn))
            cached.append((group, group_cache))

        # Polyak step in the SCION dual norm. Global: uses (Σ dn_l)². Per-layer:
        # each parameter gets polyak_l = 2f / dn_l².
        dual_norm_sq = sum(dn_list) ** 2
        if per_layer:
            polyak_global = None
        else:
            polyak_global = self.polyak_multiplier * 2.0 * loss / max(dual_norm_sq, 1e-12)

        # Pass 2: combine cfg.lr with polyak and apply update.
        update_sq, lr, lr_eff = 0.0, 0.0, 0.0
        for group, group_cache in cached:
            lr = group['lr']
            scale = group['scale']
            unconstrained = group['unconstrained']
            weight_decay = group.get('weight_decay', 0.0)
            polyak_mult_l = group.get('polyak_multiplier', self.polyak_multiplier)

            for p, lmo_g, dn in group_cache:
                if lmo_g is None:
                    continue

                if per_layer:
                    if self.polyak_form == 'lin_sdn':
                        denom = scale * dn                       # 2f / (scale · dn) — Frank-Wolfe / Demyanov-Rubinov short step
                    elif self.polyak_form == 'sqr_sdn':
                        denom = (scale * dn) ** 2                # 2f / (scale · dn)² — SGD-Polyak with scale correction
                    else:                                        # 'sqr_dn'
                        denom = dn ** 2                          # 2f / dn² — original SCION-NGN, no scale correction
                    polyak_l = polyak_mult_l * 2.0 * loss / max(denom, 1e-12)
                else:
                    polyak_l = polyak_global

                if lr <= 0 or polyak_l <= 0:
                    lr_eff = lr
                elif mode == 'ngn':
                    lr_eff = lr * polyak_l / (lr + polyak_l)    # harmonic mean
                else:                                           # 'sps'
                    lr_eff = min(lr, polyak_l)                  # stochastic Polyak

                update_sq += (lr_eff * scale) ** 2 * lmo_g.float().pow(2).sum().item()
                if weight_decay != 0:
                    p.data.mul_(1 - lr_eff * weight_decay)
                if unconstrained:
                    p.data.add_(lmo_g, alpha=-lr_eff * scale)
                else:
                    p.data.mul_(1 - lr_eff).add_(lmo_g, alpha=-lr_eff * scale)

        dampening = (lr / lr_eff) if lr_eff > 0 else 1.0
        self._log_diagnostics(lr_eff, dn_list, grad_sq, update_sq,
                              dampening=dampening, dual_norm_sq=dual_norm_sq,
                              polyak=polyak_global)

    # ------------------------------------------------------------------------
    # Diagnostics.
    # ------------------------------------------------------------------------

    def _log_diagnostics(self, lr_eff, dn_list, grad_sq, update_sq,
                         dampening, dual_norm_sq, polyak):
        L = len(dn_list)
        dn_t = torch.tensor(dn_list) if L > 0 else torch.zeros(1)
        # Use sum(d_l)^2 for the global SCION dual norm. Standard mode falls back
        # to sum(d_l^2) since it has no notion of a global Polyak.
        if dual_norm_sq is None:
            dual_norm_sq = sum(d ** 2 for d in dn_list)
        self.diagnostics = {
            'optim/lr_eff': lr_eff,
            'optim/grad_norm': grad_sq ** 0.5,
            'optim/update_norm': update_sq ** 0.5,
            'optim/dampening': dampening,
            'optim/dual_norm_sq': dual_norm_sq,
            'optim/dn_layer_std': dn_t.std().item() if L > 1 else 0.0,
            'optim/dn_layer_max': dn_t.max().item() if L > 0 else 0.0,
        }
        if polyak is not None:
            self.diagnostics['optim/polyak'] = polyak
