"""Minimal Qwen SFT run for an optimizer comparison.

Full fine-tune of Qwen2.5-0.5B on Alpaca, logging held-out loss. One (optim, lr)
per invocation; a launcher sweeps the grid. The SCION optimizer and its per-layer
param groups are reused verbatim from the pretraining code, so the same
G-NGN-Scion(alpha=7e-2) recipe is exercised on a fine-tuning task.

    python qwen_ft/finetune.py --optim gngn --lr 3e-4
    python qwen_ft/finetune.py --optim adamw --lr 1e-5
    python qwen_ft/finetune.py --optim scion --lr 1e-3
"""
import argparse
import json
import math
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optim.scion_adaptive import Scion
from optim.muonmax_momo import MuonMaxMomo
from optim.init_optim import _build_scion_param_groups

MODEL = 'Qwen/Qwen2.5-0.5B'
MAX_LEN = 512
BATCH = 8
STEPS = 600
EVAL_EVERY = 50
N_VAL = 256
WARMUP_FRAC = 0.1          # linear warmup, then cosine decay to 0 — standard SFT recipe

PROMPT = ('Below is an instruction that describes a task. '
          'Write a response that appropriately completes the request.\n\n'
          '### Instruction:\n{instruction}\n\n{maybe_input}### Response:\n')


class Cfg:
    """Duck-typed config so _build_scion_param_groups can read attributes."""
    scion_norm_steps = 5
    matrix_scale = 4.0
    embed_tokens_scale = 64.0
    lm_head_scale = 4096.0
    oned_params_scale = 8.0
    scion_unconstrained = False


def format_example(ex, tok):
    inp = ex.get('input', '').strip()
    maybe = f'### Input:\n{inp}\n\n' if inp else ''
    text = PROMPT.format(instruction=ex['instruction'].strip(), maybe_input=maybe)
    text += ex['output'].strip() + tok.eos_token
    ids = tok(text, truncation=True, max_length=MAX_LEN, return_tensors='pt')['input_ids'][0]
    return ids


def collate(batch, pad_id):
    maxlen = max(len(x) for x in batch)
    input_ids, labels = [], []
    for ids in batch:
        pad = maxlen - len(ids)
        input_ids.append(torch.cat([ids, torch.full((pad,), pad_id)]))
        labels.append(torch.cat([ids, torch.full((pad,), -100)]))  # ignore pad in loss
    return torch.stack(input_ids), torch.stack(labels)


def build_optimizer(name, model, lr, alpha=7e-2, scale_mult=1.0, scales=None,
                    muon_lr_scale=10.0, muon_momentum=0.95, wd=0.0):
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                 weight_decay=0.0, eps=1e-8)
    if name == 'sfadamw':   # Schedule-Free AdamW (Defazio): LR needed, schedule not
        import schedulefree
        return schedulefree.AdamWScheduleFree(
            model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0,
            warmup_steps=int(0.1 * STEPS))
    if name == 'psf':       # Prodigy + Schedule-Free: neither LR nor schedule
        from prodigyplus import ProdigyPlusScheduleFree
        return ProdigyPlusScheduleFree(model.parameters(), lr=lr,
                                       weight_decay=0.0)
    if name == 'sfplus':    # ScheduleFree+ (Defazio 2026), official reference impl
        from optim.adamc_schedulefree_plus_paper import AdamCScheduleFreePlusPaper
        return AdamCScheduleFreePlusPaper(model.parameters(), lr=lr,
                                          weight_decay=wd)
    if name == 'ngnmdv1':   # NGN mirror-descent, same construction as pretraining
        from optim.NGNMDv1 import NGN_MDv1
        return NGN_MDv1(model.parameters(), lr=lr, betas=(0.9, 0.95),
                        eps=1e-8, weight_decay=0.0)
    cfg = Cfg()
    if scales is not None:  # explicit (matrix, oned, embed, lm_head) override
        cfg.matrix_scale, cfg.oned_params_scale, \
            cfg.embed_tokens_scale, cfg.lm_head_scale = scales
    # scale_mult shrinks the per-group LMO scales; finetuning a pretrained model
    # can want gentler steps than pretraining, and the scale sets the step size.
    cfg.matrix_scale *= scale_mult
    cfg.embed_tokens_scale *= scale_mult
    cfg.lm_head_scale *= scale_mult
    cfg.oned_params_scale *= scale_mult
    groups = _build_scion_param_groups(model, cfg)
    if name == 'scion':
        return Scion(groups, lr=lr, momentum=0.1, ngn=False, cap='hm', form='constrained')
    if name == 'gngn':  # G-NGN-Scion, single global alpha
        for g in groups:
            g['polyak_multiplier'] = alpha
        return Scion(groups, ngn='global', cap='hm', form='constrained',
                     lr=lr, momentum=0.1)
    if name == 'muonmax':  # Muonmax-Momo, pretraining defaults
        excluded = ('embed_tokens', 'lm_head', 'wte', 'wpe')
        muon_p, adam_p = [], []
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (muon_p if p.ndim >= 2 and not any(k in pname.lower() for k in excluded)
             else adam_p).append(p)
        return MuonMaxMomo(muon_p, adam_p, lr=lr, muon_lr_scale=muon_lr_scale, wd=0.0,
                           momentum=muon_momentum, ns_steps=5, betas=(muon_momentum, 0.95),
                           eps=1e-8, truncate_loss=0.0, stale_nuc=True)
    raise ValueError(name)


def lr_scale(step, total):
    """Linear warmup for WARMUP_FRAC of the run, then cosine decay to 0."""
    warm = max(1, int(WARMUP_FRAC * total))
    if step < warm:
        return step / warm
    prog = (step - warm) / max(1, total - warm)
    return 0.5 * (1.0 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    tot, n = 0.0, 0
    for input_ids, labels in val_loader:
        out = model(input_ids=input_ids.to(device), labels=labels.to(device))
        tot += out.loss.item(); n += 1
    model.train()
    return tot / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--optim', required=True,
                   choices=['gngn', 'scion', 'adamw', 'muonmax', 'sfadamw', 'psf',
                            'sfplus', 'ngnmdv1'])
    p.add_argument('--model', default=MODEL, help='HF model id to fine-tune')
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--alpha', type=float, default=7e-2, help='G-NGN Polyak multiplier')
    p.add_argument('--scale-mult', type=float, default=1.0, help='shrink LMO scales')
    p.add_argument('--scales', default=None,
                   help='override per-group scales: "matrix,oned,embed,lm_head"')
    p.add_argument('--lora', action='store_true', help='LoRA fine-tune (r=16)')
    p.add_argument('--muon-lr-scale', type=float, default=10.0)
    p.add_argument('--muon-momentum', type=float, default=0.95)
    p.add_argument('--wd', type=float, default=0.0, help='sfplus AdamC decay')
    p.add_argument('--sfplus-warmup', type=int, default=0,
                   help='linear warmup steps on the sfplus lr multiplier')
    p.add_argument('--tag', default=None, help='results subdir override')
    p.add_argument('--out', default='qwen_ft/results_const')
    p.add_argument('--schedule', default='constant', choices=['constant', 'cosine'],
                   help='LR schedule for non-schedule-free optimizers')
    args = p.parse_args()

    device = 'cuda'
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable()
    if args.lora:
        from peft import LoraConfig, get_peft_model
        model.enable_input_require_grads()   # needed with gradient checkpointing
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                            'gate_proj', 'up_proj', 'down_proj']))
    model.train()

    ds = load_dataset('tatsu-lab/alpaca', split='train')
    ds = ds.shuffle(seed=0)
    val_raw = [format_example(ds[i], tok) for i in range(N_VAL)]
    train_raw = [format_example(ds[i], tok) for i in range(N_VAL, N_VAL + BATCH * STEPS)]

    pad_id = tok.pad_token_id
    train_loader = DataLoader(train_raw, batch_size=BATCH, shuffle=True,
                              collate_fn=lambda b: collate(b, pad_id))
    val_loader = DataLoader(val_raw, batch_size=BATCH, shuffle=False,
                            collate_fn=lambda b: collate(b, pad_id))

    scales = tuple(float(x) for x in args.scales.split(',')) if args.scales else None
    opt = build_optimizer(args.optim, model, args.lr,
                          alpha=args.alpha, scale_mult=args.scale_mult,
                          scales=scales, muon_lr_scale=args.muon_lr_scale,
                          muon_momentum=args.muon_momentum, wd=args.wd)
    # step(loss=...) optimizers: the loss value enters the step size directly
    is_scion = args.optim in ('gngn', 'scion', 'muonmax', 'ngnmdv1')

    schedule_free = args.optim in ('sfadamw', 'psf', 'sfplus')
    if schedule_free:
        opt.train()
    history, step = [], 0
    for input_ids, labels in train_loader:
        if step >= STEPS:
            break
        if not schedule_free and args.schedule == 'cosine':
            # Warmup + cosine schedule, applied by scaling each group's base LR.
            scale = lr_scale(step, STEPS)
            for g in opt.param_groups:
                g['lr'] = args.lr * scale
        elif args.sfplus_warmup > 0:
            # SF+ integrates warmup expressed through the lr sequence.
            scale = min(1.0, (step + 1) / args.sfplus_warmup)
            for g in opt.param_groups:
                g['lr'] = args.lr * scale
        out = model(input_ids=input_ids.to(device), labels=labels.to(device))
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.optim == 'sfplus':
            opt.step_func(function_value=loss.item())
        elif is_scion:
            opt.step(loss=loss.item())
        else:
            opt.step()
        step += 1
        if step % EVAL_EVERY == 0 or step == STEPS:
            if schedule_free:
                opt.eval()
            vl = evaluate(model, val_loader, device)
            if schedule_free:
                opt.train()
            history.append({'step': step, 'train_loss': loss.item(), 'val_loss': vl})
            print(f'[{args.optim} lr={args.lr:.0e}] step {step:4d}  '
                  f'train {loss.item():.4f}  val {vl:.4f}', flush=True)
            if not math.isfinite(vl) or vl > 20:
                print('diverged, stopping', flush=True)
                break

    lr_str = f'{args.lr:.0e}'.replace('e-0', 'e-').replace('e+0', 'e')
    out_dir = os.path.join(args.out, args.tag or args.optim, f'lr_{lr_str}')
    os.makedirs(out_dir, exist_ok=True)
    final = history[-1]['val_loss'] if history else float('nan')
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump({'optim': args.optim, 'lr': args.lr,
                   'final_val_loss': final, 'history': history}, f, indent=2)
    print(f'saved {out_dir}/metrics.json  final_val={final:.4f}', flush=True)


if __name__ == '__main__':
    main()
