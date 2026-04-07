"""
Sweep runner that creates individual runs with LR in folder/wandb names.

Experiment structure:
  experiments/{method}/lr_{lr_str}/metrics.json

Wandb run names:
  {method}_lr_{lr_str}

Usage:
  python sweep_run.py --config config/sweep_standard.yaml
  python sweep_run.py --config config/sweep_standard.yaml --lr 3e-4
  python sweep_run.py --all
"""
import yaml, os, sys, subprocess, argparse


def lr_to_str(lr):
    """Convert LR float to clean string: 1e-4 -> '1e-4', 3e-4 -> '3e-4'"""
    s = f'{lr:.0e}'  # e.g. '3e-04'
    # clean up: '3e-04' -> '3e-4'
    base, exp = s.split('e')
    exp = str(int(exp))
    return f'{base}e{exp}'


def run_single(base_config_path, lr, dry_run=False):
    """Run a single training job with given LR."""
    with open(base_config_path) as f:
        cfg = yaml.safe_load(f)

    method = cfg['exp_name']
    lr_str = lr_to_str(lr)

    # Override LR to scalar
    cfg['lr'] = lr

    # Set experiment name with LR
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

    # Write temp config
    tmp_cfg_path = f'/tmp/sweep_{method}_lr_{lr_str}.yaml'
    with open(tmp_cfg_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # Run training (no job_idx!)
    result = subprocess.run(
        ['python', 'train.py', '--config', tmp_cfg_path],
        cwd='/home/arman/plainLM'
    )

    os.remove(tmp_cfg_path)
    return result.returncode == 0


def run_config(config_path, specific_lr=None, dry_run=False):
    """Run all LRs for a config, or a specific LR."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    lrs = cfg.get('lr', [])
    if not isinstance(lrs, list):
        lrs = [lrs]

    if specific_lr is not None:
        lrs = [specific_lr]

    method = cfg['exp_name']
    print(f'\n{"="*50}')
    print(f'  {method} ({len(lrs)} LRs)')
    print(f'{"="*50}')

    for lr in lrs:
        run_single(config_path, lr, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Single config to run')
    parser.add_argument('--lr', type=float, help='Specific LR to run')
    parser.add_argument('--all', action='store_true', help='Run all sweep configs')
    parser.add_argument('--dry-run', action='store_true', help='Just print what would run')
    args = parser.parse_args()

    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', os.path.expanduser('~/.cache/torch_inductor'))
    os.makedirs(os.environ['TORCHINDUCTOR_CACHE_DIR'], exist_ok=True)

    if args.all:
        import glob
        configs = sorted(glob.glob('/home/arman/plainLM/config/sweep_*.yaml'))
        for cfg_path in configs:
            run_config(cfg_path, dry_run=args.dry_run)
    elif args.config:
        run_config(args.config, specific_lr=args.lr, dry_run=args.dry_run)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
