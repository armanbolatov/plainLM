"""Implements NGN-MDv1 opitmizer."""

from typing import Callable, Iterable, Optional, Tuple, Union, Any, Dict
import torch


class NGN_MDv1(torch.optim.Optimizer):
    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: Union[float, torch.Tensor] = 1.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay=0.0
        ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "wd": weight_decay}
        super().__init__(params, defaults)
        self.state["num_steps"] = 0
        self.diagnostics = {}

    @torch.no_grad()
    def step(self, closure: Callable[[], float] = None, loss: torch.Tensor = None) -> Optional[float]:
        """
        Performs a single optimization step.
          new_p = x^{t+1}
          p = x^t
          old_p = x^{t-1}

        Parameters
        ----------
        loss : torch.tensor
            The loss tensor. Use this when the backward step has already been performed. By default None.

        Returns
        -------
        (Stochastic) Loss function value.
        """
        assert (closure is not None) or (loss is not None), "Either loss tensor or closure must be passed."
        assert (closure is None) or (loss is None), "Pass either the loss tensor or the closure, not both."
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.state['num_steps'] += 1
        grad_norm_sq = self.compute_sq_grad_terms()

        raw_grad_sq = 0.0
        update_sq = 0.0

        for group in self.param_groups:
            lr = group['lr']
            beta1, _ = group['betas']
            wd = group['wd']
            dampening = 1.0 + lr * grad_norm_sq / (2 * loss)
            momngn = (1 - beta1) * lr / dampening

            for p in group['params']:
                if p.grad is None:
                  continue

                state = self.state[p]
                raw_grad_sq += p.grad.float().pow(2).sum().item()
                p_before = p.data.clone()

                if 'p' not in state:
                    state["p"] = p.detach().clone()
                    new_p = p - momngn * state["D"] * p.grad
                else:
                    old_p = state["p"]
                    state["p"] = p.detach().clone()
                    new_p = p - momngn * state["D"] * p.grad + beta1 * (p - old_p)
                    if wd > 0:
                        new_p -= lr*wd*p
                p.copy_(new_p)
                update_sq += (p.data - p_before).float().pow(2).sum().item()

        _to_float = lambda x: x.item() if torch.is_tensor(x) else float(x)
        self.diagnostics = {
            'optim/lr_eff': _to_float(momngn / (1 - group['betas'][0])) if momngn else lr,
            'optim/dampening': _to_float(dampening),
            'optim/grad_norm': raw_grad_sq ** 0.5,
            'optim/update_norm': update_sq ** 0.5,
            'optim/dual_norm_sq': _to_float(grad_norm_sq),
            'optim/dn_layer_std': 0.0,
            'optim/dn_layer_max': 0.0,
        }

        return loss

    @torch.no_grad()   
    def compute_sq_grad_terms(self):
        """Computes and returns *squared* gradient norm, updates D."""
        grad_norm_sq = 0.  # we assume all tensors are on the same device
        ns = self.state['num_steps']

        for group in self.param_groups:
            _, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                  continue
                state = self.state[p]

                if "v" not in state:
                    state["v"] = (1-beta2) * p.grad.pow(2)
                else:
                    state["v"].mul_(beta2).add_(p.grad.pow(2), alpha=(1 - beta2))

                D = 1 / (eps + torch.sqrt(state["v"] / (1 - beta2 ** ns)))
                grad_norm_sq += torch.sum(D * p.grad.pow(2))
                state["D"] = D

        return grad_norm_sq
