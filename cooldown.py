"""Run one WSD cooldown branch from a trunk fork, banking the tail checkpoints.

A horizon H is trained as: trunk up to the fork at 0.8H, then a linear decay to zero over the
remaining 0.2H steps. The last tail checkpoint is 'decay+last'.
"""

import argparse
import os
import time

import torch

from avg_utils import clean_state, load_config
from data.fineweb_bin import get_bin_dataloaders
from engine import TorchEngine
from models import construct_model
from optim.lr_schedule import LinearCooldown


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--config', required=True)
  p.add_argument('--lr', type=float, default=None)
  p.add_argument('--fork', required=True)
  p.add_argument('--steps', type=int, required=True, help='cooldown length')
  p.add_argument('--out_dir', required=True)
  p.add_argument('--ckpt_every', type=int, default=50)
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
  os.makedirs(args.out_dir, exist_ok=True)

  trainloader, _ = get_bin_dataloaders(cfg)
  model, _ = construct_model(cfg)
  engine = TorchEngine(model, cfg, args.device, 0, None)

  ckpt = torch.load(args.fork, map_location=args.device, weights_only=False)
  model.load_state_dict({k: v.to(args.device) for k, v in ckpt['model'].items()})
  engine.optimizer.load_state_dict(ckpt['opt'])
  engine.scheduler = LinearCooldown(engine.optimizer, lr_max=cfg.lr, lr_end=0.0,
                                    cooldown_start_step=0, cooldown_steps=args.steps)

  print(f'cooldown: {args.steps} steps from {args.fork}', flush=True)
  t0 = time.time()
  torch.save({'step': 0, 'lr': cfg.lr, 'model': clean_state(model, to_cpu=True)}, f'{args.out_dir}/cd_0.pt')
  for batch in trainloader:
    engine.step(batch)
    if engine.micro_steps % accum:
      continue
    step = engine.micro_steps // accum
    lr = engine.optimizer.param_groups[0]['lr']
    if step % args.ckpt_every == 0 or step == args.steps:
      torch.save({'step': step, 'lr': lr, 'model': clean_state(model, to_cpu=True)},
                 f'{args.out_dir}/cd_{step}.pt')
      print(f'  {step}/{args.steps} lr={lr:.2e} ({time.time() - t0:.0f}s)', flush=True)
    if step >= args.steps:
      break


if __name__ == '__main__':
  main()
