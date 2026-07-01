"""SCION optimizer (Spectral Conditioning by Iterative Normalisation).

Three orthogonal axes select the variant:

    ngn   : False | 'local' | 'global'
              False    -> plain SCION (no NGN adaptation)
              'local'  -> per-layer Polyak rho_l = alpha_l * 2f / (s_l * d_l)
              'global' -> single global Polyak rho = alpha_l * 2f / D,
                          where D = sum_l (s_l * d_l) is the scale-consistent
                          SCION dual norm

    form  : 'constrained' | 'regularized'   (ignored when ngn=False)
              constrained  -> dp = -gamma * s_l * LMO(m_l)
              regularized  -> dp = -gamma * s_l * factor * LMO(m_l)
                              factor = s_l * d_l  (local) or D (global)

    cap   : 'hm' | 'min'    (ignored when ngn=False)
              hm   -> gamma = (eta * rho) / (eta + rho)   (harmonic mean)
              min  -> gamma = min(eta, rho)                (SPS-style)

User-facing variants (how configs name them):

    uScion              ngn=False, unconstrained=True
    L-NGN-Scion-CSD     ngn='local',  form='constrained',  unconstrained=True
    L-NGN-Scion-RSD     ngn='local',  form='regularized',  unconstrained=True
    G-NGN-Scion-CSD     ngn='global', form='constrained',  unconstrained=True
    G-NGN-Scion-RSD     ngn='global', form='regularized',  unconstrained=True

Notation:
    g_l       raw gradient at layer l
    m_l       momentum buffer at layer l
    eta       scheduler LR at this step
    s_l       per-group LMO scale       (param-group 'scale')
    alpha_l   per-group Polyak mult     (param-group 'polyak_multiplier')
    d_l       <g_l, LMO_l(g_l)>         (per-layer dual-norm contribution)
    D         sum_l (s_l * d_l)         (scale-consistent global dual norm)
    f         minibatch loss

The dual norm is computed on the raw gradient; momentum is only used in the
update direction. `unconstrained=True` drops the (1-gamma)*p shrinkage so the
step becomes plain p <- p - gamma * s_l * factor * LMO(m_l).
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

_NGN_VALID = (False, 'local', 'global')
_FORM_VALID = ('constrained', 'regularized')
_CAP_VALID = ('hm', 'min')
_D_NORM_VALID = (None, 'd_out', 'sqrt_N', 'N')


class Scion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=1.0, norm='Spectral',
                 norm_kwargs=None, scale=1.0, unconstrained=False, weight_decay=0.0,
                 polyak_multiplier=1.0, ngn=False, form='constrained', cap='hm',
                 polyak_d_norm=None):
        if lr < 0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0:
            raise ValueError(f'Invalid momentum: {momentum}')
        if ngn not in _NGN_VALID:
            raise ValueError(f"ngn must be one of {_NGN_VALID}, got {ngn!r}")
        if form not in _FORM_VALID:
            raise ValueError(f"form must be one of {_FORM_VALID}, got {form!r}")
        if cap not in _CAP_VALID:
            raise ValueError(f"cap must be one of {_CAP_VALID}, got {cap!r}")
        if polyak_d_norm not in _D_NORM_VALID:
            raise ValueError(f"polyak_d_norm must be one of {_D_NORM_VALID}, got {polyak_d_norm!r}")
        if norm_kwargs is None:
            norm_kwargs = {}

        defaults = dict(lr=lr, momentum=momentum, norm=norm, norm_kwargs=norm_kwargs,
                        scale=scale, unconstrained=unconstrained, weight_decay=weight_decay,
                        polyak_multiplier=polyak_multiplier)
        self.polyak_d_norm = polyak_d_norm
        self.ngn = ngn
        self.form = form
        self.cap = cap
        self.polyak_multiplier = polyak_multiplier
        self.diagnostics = {}
        super().__init__(params, defaults)

    # ------------------------------------------------------------------------
    # Public dispatch.
    # ------------------------------------------------------------------------

    def step(self, loss=None):
        if self.ngn:
            self._step_ngn(loss)
        else:
            self._step_standard()

    # ------------------------------------------------------------------------
    # Plain SCION (no NGN adaptation).
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
    # NGN step (local or global; constrained or regularized; HM or min cap).
    # ------------------------------------------------------------------------

    def _step_ngn(self, loss):
        if loss is None:
            raise ValueError(f"ngn={self.ngn!r} requires loss. Pass loss=val to step().")

        # Pass 1: per-layer d_l on raw grad, update momentum buffers, cache LMO(m).
        dn_list, grad_sq = [], 0.0
        cached = []
        for group in self.param_groups:
            momentum = group['momentum']
            scale = group['scale']
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            group_cache = []
            for p in group['params']:
                g = p.grad
                if g is None:
                    group_cache.append((p, None, 0.0))
                    continue
                state = self.state[p]

                grad_sq += g.float().pow(2).sum().item()
                d_l = (g * norm_backend.lmo(g)).sum().item()
                dn_list.append(scale * d_l)          # contribution to D = sum s_l * d_l

                if momentum != 1:
                    buf = state.setdefault('momentum_buffer', torch.zeros_like(g))
                    buf.mul_(1 - momentum).add_(g, alpha=momentum)
                    g = buf

                group_cache.append((p, norm_backend.lmo(g), d_l))
            cached.append((group, group_cache))

        # Global dual norm. D uses s_l * d_l (scale-consistent for the Polyak
        # denominator, so each layer contributes in proportion to its step size).
        # D_bare uses bare d_l, used as the regularized-form factor — including
        # s_l here would double-count the LMO scale and blow up updates at
        # large-scale groups (lm_head s=4096 gives s_l^2 = 1.7e7 in the update).
        D = sum(dn_list)
        D_bare = sum(d_l for _, group_cache in cached
                     for _, lmo_m, d_l in group_cache if lmo_m is not None)

        # Pass 2: per-param rho, gamma, and update.
        group_polyak, group_lr_eff = {}, {}
        update_sq, lr, lr_eff = 0.0, 0.0, 0.0
        for group, group_cache in cached:
            lr = group['lr']
            scale = group['scale']
            unconstrained = group['unconstrained']
            weight_decay = group.get('weight_decay', 0.0)
            alpha_l = group.get('polyak_multiplier', self.polyak_multiplier)
            # polyak_skip: if True, the per-group NGN cap is disabled (gamma=lr).
            # Used to isolate "L-NGN cap only on matrix layers" experiments.
            polyak_skip = group.get('polyak_skip', False)
            # polyak_d_norm: normalize d_l (per layer) before using in the Polyak
            # denominator. Choices: None, 'd_out', 'sqrt_N', 'N'. Lets a single
            # global alpha balance per-layer dynamics that otherwise differ wildly
            # because of Sign-LMO L1 dual norms vs Spectral-LMO nuclear norms.
            polyak_acc, eff_acc, count = 0.0, 0.0, 0
            for p, lmo_m, d_l in group_cache:
                if lmo_m is None:
                    continue

                # Normalize d_l per the chosen scheme (default: no normalization).
                if self.polyak_d_norm == 'd_out' and p.ndim >= 1:
                    d_l_eff = d_l / max(p.shape[0], 1)
                elif self.polyak_d_norm == 'sqrt_N':
                    d_l_eff = d_l / max(p.numel() ** 0.5, 1)
                elif self.polyak_d_norm == 'N':
                    d_l_eff = d_l / max(p.numel(), 1)
                else:
                    d_l_eff = d_l

                # Polyak step. Linear (Demyanov–Rubinov) form, scale-consistent.
                if polyak_skip:
                    # Disable the cap for this group: behaves like plain Scion.
                    rho = float('inf')
                    reg_factor = d_l
                elif self.ngn == 'global':
                    rho = alpha_l * 2.0 * loss / max(D, 1e-12)
                    reg_factor = D_bare           # bare dual norm sum (no s_l)
                else:  # 'local'
                    rho = alpha_l * 2.0 * loss / max(scale * d_l_eff, 1e-12)
                    reg_factor = d_l              # bare layer dual norm (no s_l)

                # Effective LR via HM or min cap.
                if lr <= 0 or rho <= 0 or rho == float('inf'):
                    gamma = lr
                elif self.cap == 'hm':
                    gamma = lr * rho / (lr + rho)
                else:                                  # 'min'
                    gamma = min(lr, rho)
                lr_eff = gamma

                # CSD: gamma * s_l in the LMO direction (s_l is the per-layer
                # LMO scale, hand-tuned). RSD: gamma * ||g||_* — the dual norm
                # already carries the gradient magnitude, no extra s_l (else
                # lm_head with s=4096 + d_l~25k would blow up the update by 5
                # orders of magnitude).
                if self.form == 'regularized':
                    step_coef = gamma * reg_factor
                else:
                    step_coef = gamma * scale

                polyak_acc += rho
                eff_acc += gamma
                count += 1

                update_sq += step_coef ** 2 * lmo_m.float().pow(2).sum().item()
                if weight_decay != 0:
                    p.data.mul_(1 - gamma * weight_decay)
                if unconstrained:
                    p.data.add_(lmo_m, alpha=-step_coef)
                else:
                    p.data.mul_(1 - gamma).add_(lmo_m, alpha=-step_coef)
            if count > 0:
                group_polyak[scale] = polyak_acc / count
                group_lr_eff[scale] = eff_acc / count

        dampening = (lr / lr_eff) if lr_eff > 0 else 1.0
        self._log_diagnostics(lr_eff, dn_list, grad_sq, update_sq,
                              dampening=dampening, dual_norm_sq=D ** 2,
                              polyak=None,
                              group_polyak=group_polyak, group_lr_eff=group_lr_eff, lr=lr,
                              global_dual_norm=D if self.ngn == 'global' else None)

    # ------------------------------------------------------------------------
    # Diagnostics.
    # ------------------------------------------------------------------------

    def _log_diagnostics(self, lr_eff, dn_list, grad_sq, update_sq,
                         dampening, dual_norm_sq, polyak,
                         group_polyak=None, group_lr_eff=None, lr=None,
                         global_dual_norm=None):
        L = len(dn_list)
        dn_t = torch.tensor(dn_list) if L > 0 else torch.zeros(1)
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
        if global_dual_norm is not None:
            self.diagnostics['optim/global_dual_norm'] = global_dual_norm
        if group_polyak is not None:
            for s in sorted(group_polyak.keys()):
                tag = f'g{int(s)}' if float(s).is_integer() else f'g{s}'
                self.diagnostics[f'optim/polyak_{tag}'] = group_polyak[s]
                if group_lr_eff is not None and s in group_lr_eff:
                    self.diagnostics[f'optim/lr_eff_{tag}'] = group_lr_eff[s]
                    if lr is not None and group_lr_eff[s] > 0:
                        self.diagnostics[f'optim/damp_{tag}'] = lr / group_lr_eff[s]
