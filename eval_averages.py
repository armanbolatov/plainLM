"""Evaluate every averaging scheme offline from the banked checkpoints of one trajectory.

Expects <exp_dir>/trunk/{trunk_<step>.pt, fork_<step>.pt} and <exp_dir>/tail_<H>/cd_<step>.pt,
and writes a single json holding all series of the figure. No training happens here: a new
weighting is a weighted sum over stored checkpoints plus one evaluation pass.
"""

import argparse
import json
import os

import torch

from avg_utils import (WSM_CURVES, evaluate, fixed_alpha_log_w, list_ckpts, load_config,
                       matched_log_w, normalize, poly_log_w, weighted_average, wsm_merge_w)
from data.fineweb_bin import get_bin_dataloaders
from engine import TorchEngine
from models import construct_model


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--config', required=True)
  p.add_argument('--exp_dir', required=True)
  p.add_argument('--lr', type=float, required=True)
  p.add_argument('--mus', default='1,3,10')
  p.add_argument('--cs', default='2,16', help='growing-window constants, theta = c / t')
  p.add_argument('--wsm_windows', default='5,10,20,40')
  p.add_argument('--n_horizons', type=int, default=16, help='grid for the trunk-average curves')
  p.add_argument('--cooldown_frac', type=float, default=0.2)
  p.add_argument('--device', default='cuda:0')
  p.add_argument('--out', required=True)
  return p.parse_args()


def trunk_curves(trunk, cfg, args, mus, cs, evaluator):
  """Averages of the constant-LR run, on a grid of horizons."""
  stride = trunk[1][0] - trunk[0][0]
  first, last = trunk[0][0], trunk[-1][0]
  grid = sorted({min(trunk, key=lambda c: abs(c[0] - h))[0]
                 for h in [first + (last - first) * (i + 1) / args.n_horizons
                           for i in range(args.n_horizons)]})
  out = {}
  for horizon in grid:
    upto = [c for c in trunk if c[0] <= horizon]
    steps = [s for s, _ in upto]
    lrs = [args.lr] * len(upto)
    strides = [stride] * len(upto)
    schemes = {'uniform': [0.0] * len(upto)}
    schemes.update({f'mu_{mu:g}': matched_log_w(lrs, strides, mu) for mu in mus})
    schemes.update({f'gw_{c:g}': poly_log_w(steps, c, cfg.warmup_steps) for c in cs})

    row = {}
    for name, log_w in schemes.items():
      row[name] = evaluator(weighted_average(upto, normalize(log_w)))
    row['const_last'] = evaluator(torch.load(upto[-1][1], map_location='cpu',
                                             weights_only=False)['model'])
    out[horizon] = row
    print(f'  trunk H={horizon}: ' + ' '.join(f'{k}={v:.4f}' for k, v in row.items()), flush=True)
  return out


def wsd_points(trunk, cfg, args, mus, evaluator):
  """decay+last and the fixed-alpha EMA over the whole WSD run, per horizon."""
  stride = trunk[1][0] - trunk[0][0]
  out = {}
  for name in sorted(os.listdir(args.exp_dir)):
    if not name.startswith('tail_'):
      continue
    horizon = int(name.split('_')[1])
    fork = round((1 - args.cooldown_frac) * horizon)
    tail = [c for c in list_ckpts(f'{args.exp_dir}/{name}/cd_*.pt') if c[0] > 0]
    tail_stride = tail[1][0] - tail[0][0]
    cool = tail[-1][0]

    items = [c for c in trunk if cfg.warmup_steps <= c[0] <= fork]
    steps = [s for s, _ in items] + [fork + s for s, _ in tail]
    lrs = [args.lr] * len(items) + [args.lr * (1 - s / cool) for s, _ in tail]
    strides = [stride] * len(items) + [tail_stride] * len(tail)
    items = items + tail

    row = {'decay_last': evaluator(torch.load(tail[-1][1], map_location='cpu',
                                              weights_only=False)['model'])}
    for mu in mus:
      log_w = fixed_alpha_log_w(steps, strides, mu * args.lr)
      row[f'ema_{mu:g}'] = evaluator(weighted_average(items, normalize(log_w)))
    out[horizon] = row
    print(f'  wsd H={horizon}: ' + ' '.join(f'{k}={v:.4f}' for k, v in row.items()), flush=True)
  return out


def wsm_merges(trunk, args, windows, horizons, evaluator):
  """Windowed checkpoint merges of Tian et al. (2025), for reference."""
  out = {}
  for horizon in horizons:
    upto = [c for c in trunk if c[0] <= horizon]
    row = {}
    for curve in WSM_CURVES:
      for k in windows:
        if k >= len(upto):
          continue
        window = upto[-(k + 1):]
        row[f'{curve}_n{k}'] = evaluator(weighted_average(window, wsm_merge_w(curve, k)))
    out[horizon] = row
    best = min(row, key=row.get)
    print(f'  wsm H={horizon}: best {best}={row[best]:.4f}', flush=True)
  return out


def main():
  args = parse_args()
  cfg = load_config(args.config, lr=args.lr)
  cfg.resume = False
  mus = [float(x) for x in args.mus.split(',')]
  cs = [float(x) for x in args.cs.split(',')]
  windows = [int(x) for x in args.wsm_windows.split(',')]

  trunk = [c for c in list_ckpts(f'{args.exp_dir}/trunk/trunk_*.pt') if c[0] >= cfg.warmup_steps]
  _, validloader = get_bin_dataloaders(cfg)
  model, _ = construct_model(cfg)
  engine = TorchEngine(model, cfg, args.device, 0, None)
  eval_model, _ = construct_model(cfg)
  eval_model = eval_model.to(args.device)
  evaluator = lambda state: evaluate({k: v.float() for k, v in state.items()},
                                     eval_model, engine, validloader)

  curves = trunk_curves(trunk, cfg, args, mus, cs, evaluator)
  wsd = wsd_points(trunk, cfg, args, mus, evaluator)
  wsm = wsm_merges(trunk, args, windows, sorted(wsd), evaluator)

  tokens_per_step = cfg.seq_len * cfg.micro_batch_size * cfg.grad_accumulation_steps
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  with open(args.out, 'w') as f:
    json.dump({'lr': args.lr, 'mus': mus, 'cs': cs, 'tokens_per_step': tokens_per_step,
               'curves': curves, 'wsd': wsd, 'wsm': wsm}, f)
  print(f'saved {args.out}', flush=True)


if __name__ == '__main__':
  main()
