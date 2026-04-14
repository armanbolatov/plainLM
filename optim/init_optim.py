"""Intialize optimizer and scheduler."""

import torch
from .lr_schedule import WarmupCosine, WSD, WarmupConstant, LinearCooldown


def _build_scion_param_groups(model, cfg):
  """Build per-layer param groups for Scion, matching ScionVar conventions."""
  ns_steps = getattr(cfg, 'scion_norm_steps', 5)
  embed_scale = getattr(cfg, 'embed_tokens_scale', 64.0)
  matrix_scale = getattr(cfg, 'matrix_scale', 4.0)
  lm_head_scale = getattr(cfg, 'lm_head_scale', 2048.0)
  oned_scale = getattr(cfg, 'oned_params_scale', 1.0)
  unconstrained = getattr(cfg, 'scion_unconstrained', False)

  groups = []
  for name, p in model.named_parameters():
    if not p.requires_grad:
      continue

    if 'embed_tokens' in name or 'wte' in name:
      group = dict(params=[p], norm='Sign', norm_kwargs={'normalized': False},
                   scale=embed_scale, unconstrained=unconstrained)
    elif 'lm_head' in name:
      group = dict(params=[p], norm='Sign', norm_kwargs={},
                   scale=lm_head_scale, unconstrained=unconstrained)
    elif p.ndim >= 2:
      group = dict(params=[p], norm='Spectral',
                   norm_kwargs={'normalized': False, 'steps': ns_steps},
                   scale=matrix_scale, unconstrained=unconstrained)
    else:
      group = dict(params=[p], norm='Sign', norm_kwargs={'normalized': False},
                   scale=oned_scale, unconstrained=unconstrained)

    groups.append(group)

  return groups


def intialize_optimizer(param_groups, cfg, model=None):
  """
  Intialize an optimizer.
  NOTE: we pass weight_decay to optim, but it gets overwritten by the weight_decay in param_groups!
  """

  if cfg.optim == 'adamw':
    base_optim = torch.optim.AdamW(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
      fused=cfg.fused_optim,
      eps=getattr(cfg, 'eps', 1e-8),
    )
    # Wrap with diagnostics
    from .adamw_diag import wrap_with_diagnostics
    optimizer = wrap_with_diagnostics(base_optim)

  elif cfg.optim == 'nadamw':
    optimizer = torch.optim.NAdam(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
      decoupled_weight_decay=True,
      fused=cfg.fused_optim,
      eps=getattr(cfg, 'eps', 1e-8),
    )

  elif cfg.optim == 'sgd':
    optimizer = torch.optim.SGD(
      param_groups,
      lr=cfg.lr,
      momentum=cfg.beta1,
      dampening=cfg.dampening,
      weight_decay=cfg.weight_decay,
    )

  elif cfg.optim == 'signSGD':
    from .signSGD import signSGD

    optimizer = signSGD(
      param_groups,
      lr=cfg.lr,
      momentum=cfg.beta1,
      dampening=cfg.dampening,
      weight_decay=cfg.weight_decay,
    )

  elif cfg.optim == 'scion':
    from .scion_adaptive import Scion

    # Use per-layer param groups if model is provided and per-layer scales are set
    if model is not None and hasattr(cfg, 'matrix_scale'):
      scion_groups = _build_scion_param_groups(model, cfg)
    else:
      # Fallback: single norm for all params
      scion_norm = getattr(cfg, 'scion_norm', 'Spectral')
      norm_kwargs = {}
      if scion_norm == 'Spectral':
        norm_kwargs['steps'] = getattr(cfg, 'scion_norm_steps', 5)
      scion_groups = param_groups
      for g in scion_groups:
        g['norm'] = scion_norm
        g['norm_kwargs'] = norm_kwargs
        g['scale'] = getattr(cfg, 'scion_scale', 1.0)
        g['unconstrained'] = getattr(cfg, 'scion_unconstrained', False)

    optimizer = Scion(
      scion_groups,
      lr=cfg.lr,
      momentum=getattr(cfg, 'scion_momentum', cfg.beta1),
      weight_decay=cfg.weight_decay,
      adaptive=getattr(cfg, 'scion_adaptive', False),
      mean=getattr(cfg, 'scion_mean', 'HM'),
    )

  elif cfg.optim == 'muonmax_momo':
    from .muonmax_momo import MuonMaxMomo

    excluded = ('embed_tokens', 'lm_head', 'wte', 'wpe')
    muon_params = []
    adam_params = []
    for name, p in model.named_parameters():
      if not p.requires_grad:
        continue
      if p.ndim >= 2 and not any(k in name.lower() for k in excluded):
        muon_params.append(p)
      else:
        adam_params.append(p)

    optimizer = MuonMaxMomo(
      muon_params, adam_params,
      lr=cfg.lr,
      muon_lr_scale=getattr(cfg, 'muon_lr_scale', 10.0),
      wd=cfg.weight_decay,
      momentum=getattr(cfg, 'muon_momentum', 0.95),
      ns_steps=getattr(cfg, 'scion_norm_steps', 5),
      betas=[cfg.beta1, cfg.beta2],
      eps=getattr(cfg, 'eps', 1e-8),
      truncate_loss=getattr(cfg, 'truncate_loss', 0.0),
      stale_nuc=getattr(cfg, 'stale_nuc', True),
    )

  elif cfg.optim == 'ngnmdv1':
    from .NGNMDv1 import NGN_MDv1

    optimizer = NGN_MDv1(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      eps=getattr(cfg, 'eps', 1e-8),
      weight_decay=cfg.weight_decay,
    )

  elif cfg.optim == 'sfo_adamw':
    import schedulefree

    # warmup steps for schedulefree must be specified here
    warmup_steps = cfg.warmup_steps if isinstance(cfg.warmup_steps, int) else int(cfg.warmup_steps * cfg.steps_budget)
    optimizer = schedulefree.AdamWScheduleFree(
      param_groups,
      lr=cfg.lr,
      warmup_steps=warmup_steps,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
    )

  else:
    raise NotImplementedError(f'Not implemented optim: {cfg.optim}.')

  return optimizer


def initialize_scheduler(optimizer, cfg):
  if cfg.scheduler is None:
    return None

  ## Number of warmup steps
  # either specified directly (int) or as a fraction of steps_budget (float)
  if getattr(cfg, 'warmup_steps', None) is not None:
    warmup_steps = cfg.warmup_steps if isinstance(cfg.warmup_steps, int) else int(cfg.warmup_steps * cfg.steps_budget)

  ## Number of cooldown steps
  # either specified directly (int) or as a fraction of steps_budget (float)
  if getattr(cfg, 'cooldown_steps', None) is not None:
    cooldown_steps = (
      cfg.cooldown_steps if isinstance(cfg.cooldown_steps, int) else int(cfg.cooldown_steps * cfg.steps_budget)
    )

  ##Final LR of the schedule
  # either specified directly via `lr_end` or as a fraction of top lr via `lr_end_pct`
  if getattr(cfg, 'lr_end', None) is not None or getattr(cfg, 'lr_end_pct', None) is not None:
    lr_end = cfg.lr_end if (cfg.lr_end is not None) else (cfg.lr_end_pct * cfg.lr)

  if cfg.scheduler == 'warmup_cosine':
    scheduler = WarmupCosine(
      optimizer,
      lr_start=cfg.lr_start,
      lr_max=cfg.lr,
      lr_end=lr_end,
      warmup_steps=warmup_steps,
      T=cfg.steps_budget,
    )

  elif cfg.scheduler == 'wsd':
    cooldown_start_step = cfg.steps_budget - cooldown_steps
    scheduler = WSD(
      optimizer,
      lr_start=cfg.lr_start,
      lr_max=cfg.lr,
      lr_end=lr_end,
      warmup_steps=warmup_steps,
      cooldown_start_step=cooldown_start_step,
      cooldown_steps=cooldown_steps,
    )

  elif cfg.scheduler == 'warmup_constant':
    scheduler = WarmupConstant(
      optimizer,
      lr_start=cfg.lr_start,
      lr_max=cfg.lr,
      warmup_steps=warmup_steps,
    )

  elif cfg.scheduler == 'linear_cooldown':
    cooldown_start_step = cfg.resume_step
    scheduler = LinearCooldown(
      optimizer,
      lr_max=cfg.lr,
      lr_end=lr_end,
      cooldown_start_step=cooldown_start_step,
      cooldown_steps=cooldown_steps,
    )

  else:
    raise NotImplementedError(f'Not implemented scheduler: {cfg.scheduler}.')

  return scheduler
