"""Train the constant-LR (WSD stable) trunk, banking checkpoints and optimizer-state forks.

The banked checkpoints are what every averaging experiment is computed from; the forks are the
resume points for the cooldown branches (cooldown.py).
"""

import argparse
import copy
import os
import time

import torch

from avg_utils import clean_state, load_config
from data.fineweb_bin import get_bin_dataloaders
from engine import TorchEngine
from models import construct_model


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--config', required=True)
  p.add_argument('--lr', type=float, default=None)
  p.add_argument('--out_dir', required=True)
  p.add_argument('--ckpt_every', type=int, default=100)
  p.add_argument('--forks', required=True, help='steps at which to save model+optimizer')
  p.add_argument('--resume_fork', default=None, help='fork checkpoint to continue from')
  p.add_argument('--start_step', type=int, default=0, help='step the resumed fork sits at')
  p.add_argument('--device', default='cuda:0')
  return p.parse_args()


def main():
  args = parse_args()
  cfg = load_config(args.config, lr=args.lr)
  cfg.resume = False
  torch.manual_seed(cfg.seed)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  accum = cfg.grad_accumulation_steps
  forks = sorted(int(s) for s in args.forks.split(','))
  os.makedirs(args.out_dir, exist_ok=True)

  trainloader, _ = get_bin_dataloaders(cfg)
  model, _ = construct_model(cfg)
  engine = TorchEngine(model, cfg, args.device, 0, None)

  if args.resume_fork:
    ckpt = torch.load(args.resume_fork, map_location=args.device, weights_only=False)
    model.load_state_dict({k: v.to(args.device) for k, v in ckpt['model'].items()})
    engine.optimizer.load_state_dict(ckpt['opt'])
    engine.scheduler = None  # past warmup, LR stays at its peak

  print(f'trunk: {args.start_step} -> {cfg.steps_budget}, lr={cfg.lr}, forks={forks}', flush=True)
  t0 = time.time()
  for batch in _repeat(trainloader):
    engine.step(batch)
    if engine.micro_steps % accum:
      continue
    step = args.start_step + engine.micro_steps // accum
    lr = engine.optimizer.param_groups[0]['lr']

    if step % args.ckpt_every == 0:
      torch.save({'step': step, 'lr': lr, 'model': clean_state(model, to_cpu=True)},
                 f'{args.out_dir}/trunk_{step}.pt')
    if step in forks:
      torch.save({'model': clean_state(model, to_cpu=True),
                  'opt': copy.deepcopy(engine.optimizer.state_dict())},
                 f'{args.out_dir}/fork_{step}.pt')
      print(f'  fork {step} ({time.time() - t0:.0f}s)', flush=True)
    if step >= cfg.steps_budget:
      break


def _repeat(loader):
  while True:
    for batch in loader:
      yield batch


if __name__ == '__main__':
  main()
