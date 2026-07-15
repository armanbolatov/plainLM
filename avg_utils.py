"""Weight averaging: schedule-matched weights, checkpoint I/O, evaluation."""

import glob
import math
import re
from types import SimpleNamespace

import torch
import yaml

from engine.engine import _move_to_device


def load_config(path, **overrides):
  with open(path) as f:
    cfg = yaml.safe_load(f)
  cfg.update({k: v for k, v in overrides.items() if v is not None})
  return SimpleNamespace(**cfg)


def clean_state(module, to_cpu=False):
  """fp32 snapshot of the raw module; compile/DDP wrappers share the same params."""
  out = {}
  for k, v in module.state_dict().items():
    t = v.detach().float()
    out[k] = t.cpu().clone() if to_cpu else t
  return out


def list_ckpts(pattern):
  """[(step, path), ...] sorted by step, from paths ending in _<step>.pt"""
  found = [(int(re.search(r'_(\d+)\.pt$', p).group(1)), p) for p in glob.glob(pattern)]
  return sorted(found)


@torch.no_grad()
def evaluate(state, eval_model, engine, loader):
  eval_model.load_state_dict(state)
  eval_model.eval()
  total, n = 0.0, 0
  for batch in loader:
    inputs, targets, attn = _move_to_device(batch, engine.seq_len, engine.device, engine.intra_doc_masking)
    with engine.ctx:
      out = eval_model(inputs, attn)
      logits = getattr(out, 'logits', out)
      total += engine.criterion(logits.view(-1, logits.size(-1)), targets.view(-1)).item()
      n += 1
  return total / n


def weighted_average(items, weights):
  """sum_i w_i x_i over checkpoints, streamed one file at a time."""
  acc = None
  for (_, path), w in zip(items, weights):
    if w <= 1e-9:
      continue
    state = torch.load(path, map_location='cpu', weights_only=False)['model']
    if acc is None:
      acc = {k: v.float() * w for k, v in state.items()}
    else:
      for k in acc:
        acc[k].add_(state[k].float(), alpha=w)
    del state
  return acc


# --- averaging weights -------------------------------------------------------
# Checkpoints are a subsample of the iterates: the one at step t stands for the
# `stride` steps of its window, so its weight is the window's total.

def normalize(log_w):
  m = max(log_w)
  w = [math.exp(x - m) for x in log_w]
  total = sum(w)
  return [x / total for x in w]


def matched_log_w(lrs, strides, mu):
  """Two-pass weights for the effective steps theta_t = mu * lr_t (Remark 'practical recipe')."""
  log_w, prev = [], None
  for lr, stride in zip(lrs, strides):
    theta = 1 - (1 - min(mu * lr, 0.999)) ** stride
    theta = min(max(theta, 1e-12), 1 - 1e-12)
    log_w.append(0.0 if prev is None else log_w[-1] + math.log(theta / prev) - math.log(1 - theta))
    prev = theta
  return log_w


def fixed_alpha_log_w(steps, strides, alpha):
  """Constant-step optimal weights w_t ~ (1 - alpha)^{-t}, alpha = mu * lr_peak."""
  a = min(alpha, 0.999)
  return [math.log((1 - (1 - a) ** s) / a) - t * math.log(1 - a) for t, s in zip(steps, strides)]


def poly_log_w(steps, c, offset):
  """Growing window theta_t = c / t, i.e. polynomial weights w_t ~ t^(c-1)."""
  return [(c - 1) * math.log(max(t - offset + 1, 1)) for t in steps]


WSM_CURVES = {
  'linear': lambda t: 1 - t,
  'cosine': lambda t: (1 + math.cos(math.pi * t)) / 2,
  'sqrt': lambda t: 1 - math.sqrt(t),
  'ema': lambda t: 1 - 0.1 ** (1 - t),
}


def wsm_merge_w(curve, k):
  """Merge weights for the last k+1 checkpoints from a gradient decay curve (Tian et al. 2025)."""
  w = [WSM_CURVES[curve](i / k) for i in range(1, k + 1)]
  c = [1 - w[0]] + [w[j - 1] - w[j] for j in range(1, k)] + [w[k - 1]]
  total = sum(c)
  return [x / total for x in c]
