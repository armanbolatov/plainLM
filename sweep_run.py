"""
Sweep runner that creates individual runs with LR in folder/wandb names.

Experiment layout (one root per model size):
  exps_70m/
    config/_base.yaml          <- shared keys for all methods at this scale
    config/sweep_*.yaml        <- per-method overrides only (optim, scion_*, exp_name, ...)
    experiments/{method}/lr_{lr_str}/metrics.json
    plots/*.png

When loading a sweep_*.yaml, sweep_run.py merges it on top of _base.yaml from
the same directory. The final, merged config is what gets passed to train.py,
so older "self-contained" configs (without an _base.yaml) still work.

Wandb run name: {method}_lr_{lr_str}

Usage:
  python sweep_run.py                                      # run all 70m sweeps
  python sweep_run.py --exp-root exps_160m                 # run all 160m sweeps
  python sweep_run.py --config exps_70m/config/sweep_L1.yaml
  python sweep_run.py --config exps_70m/config/sweep_L1.yaml --lr 3e-4
  python sweep_run.py --dry-run
"""
import yaml, os, subprocess, argparse, glob


def lr_to_str(lr):
    """Convert LR float to clean string: 1e-4 -> '1e-4', 3e-4 -> '3e-4'"""
    s = f'{lr:.0e}'
    base, exp = s.split('e')
    exp = str(int(exp))
    return f'{base}e{exp}'


def load_config_with_base(config_path):
    """Load a sweep config, merging it on top of _base.yaml from the same dir."""
    with open(config_path) as f:
        overrides = yaml.safe_load(f)
    base_path = os.path.join(os.path.dirname(config_path), '_base.yaml')
    if os.path.exists(base_path):
        with open(base_path) as f:
            cfg = yaml.safe_load(f)
        cfg.update(overrides)
        return cfg
    return overrides


def run_single(base_config_path, lr, dry_run=False):
    """Run a single training job with given LR."""
    cfg = load_config_with_base(base_config_path)

    method = cfg['exp_name']
    lr_str = lr_to_str(lr)

    # Override LR to scalar
    cfg['lr'] = lr
    cfg['exp_name'] = f'{method}/lr_{lr_str}'
    cfg['wandb_run_name'] = f'{method}_lr_{lr_str}'

    exp_dir = os.path.join(cfg['out_dir'], cfg['exp_name'])
    metrics_path = os.path.join(exp_dir, 'metrics.json')

    if os.path.exists(metrics_path):
        print(f'  SKIP {method} lr={lr_str} (already done)')
        return True

    print(f'  RUN  {method} lr={lr_str}')
    if dry_run:
        return False

    tmp_cfg_path = f'/tmp/sweep_{method}_lr_{lr_str}.yaml'
    with open(tmp_cfg_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # Run train.py from the repo root (where this script lives), so that
    # relative paths in the config (e.g. out_dir: ./exps_410m/experiments)
    # resolve correctly regardless of the cluster path.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        ['python', 'train.py', '--config', tmp_cfg_path],
        cwd=repo_root,
    )
    os.remove(tmp_cfg_path)
    return result.returncode == 0


def run_config(config_path, specific_lr=None, dry_run=False):
    """Run all LRs for a config, or a specific LR."""
    cfg = load_config_with_base(config_path)

    lrs = cfg.get('lr', [])
    if not isinstance(lrs, list):
        lrs = [lrs]
    if specific_lr is not None:
        lrs = [specific_lr]

    method = cfg['exp_name']
    print(f'\n{"="*50}\n  {method} ({len(lrs)} LRs)\n{"="*50}')
    for lr in lrs:
        run_single(config_path, lr, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(
        description='Run a sweep of single-LR training jobs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--exp-root', type=str, default='exps_70m',
                        help='Experiment root folder (default: exps_70m). '
                             'Configs are read from {exp-root}/config/sweep_*.yaml.')
    parser.add_argument('--config', type=str, default=None,
                        help='Run only a single config file (overrides --exp-root for selection).')
    parser.add_argument('--lr', type=float, default=None,
                        help='Run only a single LR (used with --config).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just print what would run, do not actually run training.')
    args = parser.parse_args()

    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', os.path.expanduser('~/.cache/torch_inductor'))
    os.makedirs(os.environ['TORCHINDUCTOR_CACHE_DIR'], exist_ok=True)

    if args.config:
        run_config(args.config, specific_lr=args.lr, dry_run=args.dry_run)
    else:
        config_dir = os.path.abspath(os.path.join(args.exp_root, 'config'))
        if not os.path.isdir(config_dir):
            print(f'ERROR: config dir {config_dir} does not exist')
            return
        configs = sorted(glob.glob(os.path.join(config_dir, 'sweep_*.yaml')))
        # _base.yaml is not a sweep config; sweep_*.yaml glob already excludes it,
        # but be defensive in case the glob pattern is changed.
        configs = [c for c in configs if not os.path.basename(c).startswith('_')]
        if not configs:
            print(f'ERROR: no sweep_*.yaml configs found in {config_dir}')
            return
        print(f'Found {len(configs)} configs in {config_dir}')
        for cfg_path in configs:
            run_config(cfg_path, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
