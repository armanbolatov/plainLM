# muon_plainlm_single_lr.py
from __future__ import annotations

import math
from typing import Iterable, Tuple, Sequence, Optional, List, Set, Dict, Any

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer
import numpy as np

# Ignore excessive warnings
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

# WandB – Import the wandb library
import wandb

# --- Default Newton–Schulz (quintic) coefficients commonly used with Muon ---
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315


def _as_2d(t: Tensor) -> Tensor:
    """View any ndim>=2 tensor as (rows=t.shape[0], cols=rest)."""
    return t.reshape(t.shape[0], -1)


def _adjust_lr(lr: float, adjust_lr_fn: Optional[str], shape_ab: Tuple[int, int]) -> float:
    """
    Optional Muon LR adjustment. If you truly want the *same* LR behavior across Muon/AdamW,
    keep adjust_lr_fn=None.

      - None: no adjustment
      - "original": sqrt(max(1, A/B))
      - "match_rms_adamw": 0.2*sqrt(max(A,B))  (this changes Muon effective step size)
    """
    if adjust_lr_fn is None:
        return lr
    A, B = shape_ab
    if adjust_lr_fn == "original":
        return lr * math.sqrt(max(1.0, A / B))
    if adjust_lr_fn == "match_rms_adamw":
        return lr * (0.2 * math.sqrt(max(A, B)))
    raise ValueError("adjust_lr_fn must be None, 'original', or 'match_rms_adamw'")
    
    
def _spectral_norm_power(
    A2d: Tensor,
    state: Dict[str, Any],
    *,
    iters: int = 1,
    eps: float = 1e-12,
) -> Tensor:
    """
    Approximate spectral norm (largest singular value) using power iteration.
    Keeps a persistent vector in optimizer state to make it cheap per step.
    """
    A = A2d.to(torch.float32)
    m, n = A.shape

    v = state.get("spec_v", None)
    if v is None or (not torch.is_tensor(v)) or v.numel() != n or v.device != A.device:
        v = torch.randn(n, device=A.device, dtype=torch.float32)

    v = v / (v.norm() + eps)

    for _ in range(iters):
        u = A @ v
        u = u / (u.norm() + eps)
        v = A.T @ u
        v = v / (v.norm() + eps)

    state["spec_v"] = v.detach()
    sigma = (A @ v).norm()
    return sigma


def _nuclear_norm_from_ns(
    A2d: Tensor,
    *,
    ns_steps: int,
    eps: float,
    ns_dtype: torch.dtype,
    coeffs: Tuple[float, float, float],
) -> Tensor:
    """
    Approximate nuclear norm using polar factor Q ≈ UV^T:
      ||A||_* = trace(A^T Q) = <A, Q>_F  (if Q is exact polar factor)
    We compute Q via the same Newton–Schulz used in Muon.

    Note: This is still an approximation; increasing ns_steps improves accuracy.
    """
    A_f32 = A2d.to(torch.float32)
    Q = _zeropower_via_newtonschulz5(
        A_f32,
        ns_steps=ns_steps,
        eps=eps,
        ns_dtype=ns_dtype,
        coeffs=coeffs,
    )
    # trace(A^T Q) == sum(A * Q) elementwise for same-shaped matrices
    return torch.sum(A_f32 * Q.to(torch.float32)).abs()


def _zeropower_via_newtonschulz5(
    grad2d_f32: Tensor,
    *,
    ns_steps: int,
    eps: float,
    ns_dtype: torch.dtype,
    coeffs: Tuple[float, float, float],
) -> Tensor:
    """
    Muon orthogonalization: approximate nearest semi-orthogonal matrix to grad2d.

    Expects float32 input; runs iteration in ns_dtype (often bf16 on CUDA) for speed.
    """
    if grad2d_f32.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape={tuple(grad2d_f32.shape)}")
    if ns_steps <= 0 or ns_steps >= 100:
        raise ValueError("ns_steps must be in [1, 99].")

    a, b, c = coeffs

    X = grad2d_f32
    transposed = False

    # Prefer rows <= cols for cheaper Gram matrix
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    # Normalize for stability
    X = X / X.norm().clamp(min=eps)

    # Run iteration in reduced dtype if desired
    X = X.to(dtype=ns_dtype)

    for _ in range(ns_steps):
        A = X @ X.T
        # B = b*A + c*(A@A)
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        # X = a*X + (B@X)
        X = torch.addmm(X, B, X, beta=a)

    if transposed:
        X = X.T
    return X


class MyMuon(Optimizer):
    """
    Combined optimizer (single LR):
      - Muon for matrix-type params (ndim >= 2), excluding embed_tokens + lm_head by name
      - AdamW for the rest
      - Weight decay applied to ALL params (no decay/nodecay split)

    Designed for plainLM Transformer:

      self.embed_tokens = nn.Embedding(...)
      self.lm_head      = nn.Linear(...)

    Exclude keywords default to ("embed_tokens", "lm_head").

    Usage:
      opt = Muon(model.named_parameters(), lr=3e-4, weight_decay=0.1)

    Notes:
      - If tie_embeddings=True, lm_head.weight == embed_tokens.weight (same tensor).
        This optimizer de-dupes by tensor id, and routing will still send it to AdamW
        because the name "embed_tokens.weight" is excluded.
      - To keep "same LR", keep adjust_lr_fn=None (default).
    """

    def __init__(
        self,
        named_params: Iterable[Tuple[str, Parameter]],
        *,
        # shared
        lr: float = 3e-4,
        weight_decay: float = 0.1,

        # Muon hyperparams
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        ns_steps: int = 5,
        muon_eps: float = 1e-7,
        ns_dtype: torch.dtype = torch.bfloat16,
        ns_coeffs: Tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        adjust_lr_fn: Optional[str] = None,

        # AdamW hyperparams
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        use_fp32_state: bool = True,

        # routing
        excluded_keywords: Sequence[str] = ("embed_tokens", "lm_head"),

        verbose: bool = False,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be >= 0")
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if muon_momentum < 0:
            raise ValueError("muon_momentum must be >= 0")
        if muon_nesterov and muon_momentum == 0.0:
            raise ValueError("Nesterov requires muon_momentum > 0")
        if ns_steps <= 0 or ns_steps >= 100:
            raise ValueError("ns_steps must be in [1, 99]")
        if muon_eps <= 0:
            raise ValueError("muon_eps must be > 0")
        if adamw_eps <= 0:
            raise ValueError("adamw_eps must be > 0")
        if len(adamw_betas) != 2:
            raise ValueError("adamw_betas must be (beta1, beta2)")
        b1, b2 = float(adamw_betas[0]), float(adamw_betas[1])
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError("adamw_betas entries must be in [0, 1)")

        self.use_fp32_state = bool(use_fp32_state)
        self.excluded_keywords = tuple(k.lower() for k in excluded_keywords)

        # Build a single de-duped parameter list and decide method per param.
        params: List[Parameter] = []
        self.muon_param_names: List[str] = []
        self.adamw_param_names: List[str] = []
        
        
        named_params = list(named_params)

        if len(named_params) > 0 and isinstance(named_params[0], dict):
            # It's param_groups (list of dicts). Flatten group["params"] into (name, param) pairs.
            flat: list[tuple[str, torch.Tensor]] = []
            for g in named_params:
                if "params" not in g:
                    continue
                flat.extend(g["params"])
            named_params = flat
        
        # Validate that we really have (name, param) pairs now
        for item in named_params:
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                raise TypeError(
                    "Muon expected iterable of (name, param) tuples. "
                    "If you pass get_param_groups(...), you must call it with use_muon=True "
                    "so each group['params'] contains (name, param) pairs."
                )


        seen: Set[int] = set()
        name_of: Dict[int, str] = {}

        for name, p in named_params:
            if p is None or not isinstance(p, torch.Tensor):
                continue
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            name_of[pid] = name
            params.append(p)

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,

            muon_momentum=muon_momentum,
            muon_nesterov=muon_nesterov,
            ns_steps=ns_steps,
            muon_eps=muon_eps,
            ns_dtype=ns_dtype,
            ns_coeffs=ns_coeffs,
            adjust_lr_fn=adjust_lr_fn,

            betas=adamw_betas,
            eps=adamw_eps,
        )
        super().__init__(params, defaults)

        # Decide routing and store per-param method in state.
        for p in params:
            name = name_of[id(p)]
            method = "muon" if self._use_muon(name, p) else "adamw"
            self.state[p]["method"] = method
            self.state[p]["name"] = name
            if method == "muon":
                self.muon_param_names.append(name)
            else:
                self.adamw_param_names.append(name)

        if verbose:
            print(f"[Muon]  {len(self.muon_param_names)} params")
            print(f"[AdamW] {len(self.adamw_param_names)} params")
            # Uncomment if you want to see exactly which names were excluded:
            # print("AdamW names:", self.adamw_param_names)

    def _use_muon(self, name: str, p: Parameter) -> bool:
        # must be "matrix-type"
        if p.ndim < 2:
            return False
        lname = name.lower()
        if any(k in lname for k in self.excluded_keywords):
            return False
        return True

    @torch.no_grad()
    def step(self, closure=None, cur_loss=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        num = 0.0
        den = 0.0
        grad_norm = 0.0

        # single param group
        group = self.param_groups[0]
        lr = float(group["lr"])
        wd = float(group["weight_decay"])

        # muon hyperparams
        momentum = float(group["muon_momentum"])
        nesterov = bool(group["muon_nesterov"])
        ns_steps = int(group["ns_steps"])
        muon_eps = float(group["muon_eps"])
        ns_coeffs = group["ns_coeffs"]
        adjust_lr_fn = group.get("adjust_lr_fn", None)

        # choose ns_dtype safely
        requested_ns_dtype = group.get("ns_dtype", torch.bfloat16)

        # adamw hyperparams
        beta1, beta2 = group["betas"]
        beta1, beta2 = float(beta1), float(beta2)
        eps = float(group["eps"])

        for p in group["params"]:
            g = p.grad
            if g is None:
                continue
            if g.is_sparse:
                raise RuntimeError("This optimizer does not support sparse gradients.")
            if torch.is_complex(p):
                raise RuntimeError("This optimizer does not support complex parameters.")

            method = self.state[p]["method"]
            
            if 'prev_p' not in self.state[p]:
                self.state[p]['prev_p'] = p.clone().detach()
            if 'prev_grad' not in self.state[p]:
                self.state[p]['prev_grad'] = g.clone().detach()
                    
            num += torch.sum(torch.mul(self.state[p]['prev_grad']-g, self.state[p]['prev_grad']-g)).item() # ||\nabla f(x_k) - \nabla f(x_{k-1})||^2
            grad_norm += torch.sum(torch.mul(g, g)).item()
            self.state[p]['prev_grad'] = g # nabla f(x_{k-1}) <- nabla f(x_k)

            if method == "muon":
                # flatten gradient to 2D if needed
                g2d = g if g.ndim == 2 else _as_2d(g)
                g_f32 = g2d.to(torch.float32)

                st = self.state[p]
                
                
                # momentum buffer in 2D float32 space
                if momentum != 0.0:
                    buf = st.get("momentum_buffer", None)
                    if buf is None or buf.shape != g_f32.shape or buf.device != g_f32.device:
                        buf = torch.zeros_like(g_f32)
                        st["momentum_buffer"] = buf

                    buf.mul_(momentum).add_(g_f32)

                    upd = g_f32.add(buf, alpha=momentum) if nesterov else buf
                else:
                    upd = g_f32

                # ns_dtype: bf16 is great on CUDA; fall back to fp32 on CPU if needed
                ns_dtype = requested_ns_dtype
                if p.device.type != "cuda":
                    ns_dtype = torch.float32

                upd_ortho = _zeropower_via_newtonschulz5(
                    upd,
                    ns_steps=ns_steps,
                    eps=muon_eps,
                    ns_dtype=ns_dtype,
                    coeffs=ns_coeffs,
                )

                adj_lr = _adjust_lr(lr, adjust_lr_fn, upd.shape)

                # reshape back to parameter shape
                upd_full = upd_ortho.reshape_as(g2d)
                if g.ndim != 2:
                    upd_full = upd_full.reshape_as(g)
                
                den += torch.sum(torch.mul(p-st['prev_p'], p-st['prev_p'])).item()
                st['prev_p'] = p.clone().detach() # x_{k-1} <= x_k
                
                # decoupled weight decay for ALL params (Muon + AdamW)
                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)
                
                p.add_(upd_full.to(dtype=p.dtype), alpha=-adj_lr)

            else:
                # AdamW
                st = self.state[p]
                if "step" not in st:
                    st["step"] = 0
                    if self.use_fp32_state:
                        st["exp_avg"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)
                        st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)
                    else:
                        st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                st["step"] += 1
                t = st["step"]

                exp_avg = st["exp_avg"]
                exp_avg_sq = st["exp_avg_sq"]

                g_f = g.to(torch.float32) if self.use_fp32_state else g

                exp_avg.mul_(beta1).add_(g_f, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)

                # bias correction
                bc1 = 1.0 - (beta1 ** t)
                bc2 = 1.0 - (beta2 ** t)
                step_size = lr * math.sqrt(bc2) / bc1

                denom = exp_avg_sq.sqrt().add_(eps)
                update = (exp_avg / denom) * step_size
                
                den += torch.sum(torch.mul(p-st['prev_p'], p-st['prev_p'])).item()
                st['prev_p'] = p.clone().detach() # x_{k-1} <= x_k
                
                # decoupled weight decay for ALL params (Muon + AdamW)
                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)
                
                p.add_(update.to(dtype=p.dtype), alpha=-1.0)
              
        self.local_smoothness = np.sqrt(num)/(np.sqrt(den)+1e-8)
                
        wandb.log({
           "clip_grad_norm":grad_norm,
           "clip_num": np.sqrt(num),
           "clip_den": np.sqrt(den),
           "clip_local_smooth": self.local_smoothness,
           "clip_local_train_loss": cur_loss
          })
              
        return loss
        
    @torch.no_grad()
    def comp_smoothness(self, closure=None, cur_loss=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        num = 0.0
        den = 0.0
        grad_norm = 0.0
        
        group = self.param_groups[0]
        smooth_approx_norm = group.get("smooth_approx_norm", 'fro')
        
        for group in self.param_groups:
            # gather lists
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad_norm += torch.sum(torch.mul(p.grad,p.grad)).item()
                # state init
                
                if 'prev_p' not in state:
                    state['unclip_prev_grad'] = p.grad
                    state['prev_p'] = p.clone().detach()

                
                num += torch.sum(torch.mul(state['unclip_prev_grad']-p.grad, state['unclip_prev_grad']-p.grad)).item()
                state['unclip_prev_grad'] = p.grad
                den += torch.sum(torch.mul(p-state['prev_p'], p-state['prev_p'])).item()
                #state['prev_p'] = p
                
        num = np.sqrt(num)
        den = np.sqrt(den)
        self.local_smoothness = num/(den+1e-8)
                
        wandb.log({
           "unclip_num": num,
           "unclip_den": den,
           "unclip_grad_norm": np.sqrt(grad_norm),
           "unclip_local_smooth": self.local_smoothness,
           "unclip_local_train_loss": cur_loss
          })

        return loss
