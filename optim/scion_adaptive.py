import torch

def zeroth_power_via_svd(G):
   U, S, V = G.svd()
   return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Norm(object):
    def lmo(self, g):
        raise NotImplementedError


class Spectral(Norm):
    def __init__(self, normalized=True, steps=5):
        self.normalized = normalized
        self.steps = steps

    def lmo(self, g):
        if g.dim() < 2:
            fan_in = max(g.numel() // g.shape[0], 1) if g.dim() >= 1 else 1
            return (1/fan_in)*torch.sign(g)
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        d_out, d_in = g.shape
        if self.normalized:
            g *= (d_out / d_in)**0.5
        else:
            g *= d_out**0.5
        return g


class Sign(Norm):
    def __init__(self, normalized=True, zero_init=False):
        self.normalized = normalized
        self.zero_init = zero_init

    def lmo(self, g):
        if self.normalized:
            fan_in = max(g.numel() // g.shape[0], 1) if g.dim() >= 1 else 1
            return (1/fan_in)*torch.sign(g)
        return torch.sign(g)


norm_dict = {
    'Spectral': Spectral,
    'Sign': Sign
}

class Scion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=1.0, norm: str='Spectral',
                 norm_kwargs: dict=None, scale=1.0, unconstrained=False, weight_decay=0.0,
                 adaptive=False):
        """
        adaptive: False | 'per_layer' | 'L1' | 'L2' | 'L1_avg' | 'L3' | True (='per_layer')

        L3 = RSD-style: update_l = dn_l * scale * LMO(g), i.e. multiply LMO by per-layer dual norm.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if norm_kwargs is None:
            norm_kwargs = {}
        defaults = dict(lr=lr, momentum=momentum, norm=norm, norm_kwargs=norm_kwargs,
                        scale=scale, unconstrained=unconstrained, weight_decay=weight_decay)
        # normalize adaptive flag
        if adaptive is True:
            adaptive = 'per_layer'
        self.adaptive = adaptive
        self.diagnostics = {}
        super().__init__(params, defaults)

    def step(self, loss=None):
        if self.adaptive == 'per_layer':
            self._step_adaptive_per_layer(loss)
        elif self.adaptive == 'L2':
            self._step_adaptive_global(loss, mode='L2')
        elif self.adaptive == 'L1_avg':
            self._step_adaptive_global(loss, mode='L1_avg')
        elif self.adaptive == 'L1':
            self._step_adaptive_global(loss, mode='L1')
        elif self.adaptive == 'L3':
            self._step_adaptive_global(loss, mode='L3')
        elif self.adaptive:
            self._step_adaptive_per_layer(loss)
        else:
            self._step_standard()

    def _step_standard(self):
        """Original single-pass update (no adaptive step size)."""
        grad_sq = 0.0
        update_sq = 0.0
        dn_list = []
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
                dn = (g * norm_backend.lmo(g)).sum().item()
                dn_list.append(dn)

                # momentum buffer
                if momentum != 1:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(1 - momentum).add_(g, alpha=momentum)
                    g = buf

                # LMO-based update
                update = scale * norm_backend.lmo(g)
                update_sq += (lr ** 2) * update.float().pow(2).sum().item()

                # apply weight decay decoupled from gradient
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                # apply update
                if unconstrained:
                    p.data.add_(update, alpha=-lr)
                else:
                    p.data.mul_(1 - lr).add_(update, alpha=-lr)

        L = len(dn_list)
        dn_t = torch.tensor(dn_list) if L > 0 else torch.zeros(1)
        self.diagnostics = {
            'optim/lr_eff': lr,
            'optim/dual_norm_sq': sum(d ** 2 for d in dn_list),
            'optim/grad_norm': grad_sq ** 0.5,
            'optim/dampening': 1.0,
            'optim/dn_layer_std': dn_t.std().item() if L > 1 else 0.0,
            'optim/dn_layer_max': dn_t.max().item() if L > 0 else 0.0,
            'optim/update_norm': update_sq ** 0.5,
        }

    def _step_adaptive_per_layer(self, loss):
        """Per-layer adaptive: dampen lr per layer, keep scale intact."""
        if loss is None:
            raise ValueError("Adaptive mode requires loss. Pass loss=val to step().")

        grad_sq = 0.0
        update_sq = 0.0
        dn_list = []
        lr_eff_list = []
        dampening_list = []

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

                # per-layer dual norm on RAW gradient
                dn = (g * norm_backend.lmo(g)).sum().item()
                dn_list.append(dn)

                # momentum
                if momentum != 1:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(1 - momentum).add_(g, alpha=momentum)
                    g = buf

                # dampen lr, not scale: lr_eff = lr / (1 + (lr/(2f)) * dn²)
                d = 1.0 + (lr / (2.0 * loss)) * dn ** 2
                lr_eff = lr / d
                dampening_list.append(d)
                lr_eff_list.append(lr_eff)

                # update = scale * LMO(g), step = lr_eff * scale * LMO(g)
                update = norm_backend.lmo(g)
                update_sq += (lr_eff * scale) ** 2 * update.float().pow(2).sum().item()
                if weight_decay != 0:
                    p.data.mul_(1 - lr_eff * weight_decay)
                if unconstrained:
                    p.data.add_(update, alpha=-lr_eff * scale)
                else:
                    p.data.mul_(1 - lr_eff).add_(update, alpha=-lr_eff * scale)

        L = len(dn_list)
        dn_t = torch.tensor(dn_list) if L > 0 else torch.zeros(1)
        avg_lr_eff = sum(lr_eff_list) / max(L, 1)
        avg_dampening = sum(dampening_list) / max(L, 1)
        self.diagnostics = {
            'optim/lr_eff': avg_lr_eff,
            'optim/dual_norm_sq': sum(d ** 2 for d in dn_list),
            'optim/grad_norm': grad_sq ** 0.5,
            'optim/dampening': avg_dampening,
            'optim/dn_layer_std': dn_t.std().item() if L > 1 else 0.0,
            'optim/dn_layer_max': dn_t.max().item() if L > 0 else 0.0,
            'optim/update_norm': update_sq ** 0.5,
        }

    def _step_adaptive_global(self, loss, mode='L2'):
        """
        Global adaptive step-size with modes:
          'L2':     dual_norm_sq := Σ_l dn_l²          (CSD update)
          'L1':     dual_norm_sq := (Σ_l dn_l)²        (CSD update)
          'L1_avg': dual_norm_sq := (Σ_l dn_l / L)²    (CSD update)
          'L3':     dual_norm_sq := Σ_l dn_l²           (RSD update: multiply LMO by dn_l)
        """
        if loss is None:
            raise ValueError("Adaptive mode requires loss. Pass loss=val to step().")

        rsd = (mode == 'L3')  # regularized steepest descent

        # Phase 1: compute per-layer dual norms, momentum, cache updates
        dn_list = []
        grad_sq = 0.0
        cached = []
        dn_idx = 0  # track which dn corresponds to which param
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

                # dual norm on RAW gradient
                dn = (g * norm_backend.lmo(g)).sum().item()
                dn_list.append(dn)

                # momentum
                if momentum != 1:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(1 - momentum).add_(g, alpha=momentum)
                    g = buf

                update = norm_backend.lmo(g)
                group_cache.append((p, update, dn))
            cached.append((group, group_cache))

        # Phase 2: compute global dual_norm_sq
        L = len(dn_list)
        if mode == 'L2' or mode == 'L3':
            dual_norm_sq = sum(d ** 2 for d in dn_list)
        elif mode == 'L1':
            dual_norm_sq = sum(dn_list) ** 2
        else:  # L1_avg
            dual_norm_sq = (sum(dn_list) / max(L, 1)) ** 2

        # Phase 3: dampen lr globally
        update_sq = 0.0
        lr_eff = None
        dampening = 1.0
        for group, group_cache in cached:
            lr = group['lr']
            scale = group['scale']
            unconstrained = group['unconstrained']
            weight_decay = group.get('weight_decay', 0.0)

            dampening = 1.0 + (lr / (2.0 * loss)) * dual_norm_sq
            lr_eff = lr / dampening

            for p, update, dn_l in group_cache:
                if update is None:
                    continue
                # L3 (RSD): multiply update by per-layer dual norm
                effective_scale = scale * dn_l if rsd else scale

                update_sq += (lr_eff * effective_scale) ** 2 * update.float().pow(2).sum().item()
                if weight_decay != 0:
                    p.data.mul_(1 - lr_eff * weight_decay)
                if unconstrained:
                    p.data.add_(update, alpha=-lr_eff * effective_scale)
                else:
                    p.data.mul_(1 - lr_eff).add_(update, alpha=-lr_eff * effective_scale)

        dn_t = torch.tensor(dn_list) if L > 0 else torch.zeros(1)
        self.diagnostics = {
            'optim/lr_eff': lr_eff if lr_eff is not None else 0.0,
            'optim/dual_norm_sq': dual_norm_sq,
            'optim/grad_norm': grad_sq ** 0.5,
            'optim/dampening': dampening,
            'optim/dn_layer_std': dn_t.std().item() if L > 1 else 0.0,
            'optim/dn_layer_max': dn_t.max().item() if L > 0 else 0.0,
            'optim/update_norm': update_sq ** 0.5,
        }
