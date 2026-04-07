#!/bin/bash
set -e
cd "$(dirname "$0")"

# Share torch.compile cache across jobs
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torch_inductor"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

NUM_LR=8  # [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

run_sweep() {
  local name="$1"
  local config="$2"

  if [ ! -f "$config" ]; then
    echo "  SKIP $name: $config not found"
    return
  fi

  echo ""
  echo "══════════════════════════════════════════════"
  echo "  $name"
  echo "══════════════════════════════════════════════"
  for i in $(seq 0 $((NUM_LR - 1))); do
    local exp_dir
    exp_dir=$(python -c "
import yaml
with open('$config') as f: c = yaml.safe_load(f)
print(f\"{c['out_dir']}/{c['exp_name']}/job_idx_{$i}\")
" 2>/dev/null)
    if [ -f "${exp_dir}/metrics.json" ]; then
      echo "  job_idx=$i: SKIP (already done)"
      continue
    fi
    echo "  job_idx=$i: running..."
    python train.py --config "$config" --job_idx "$i" || true
  done
  echo "  ── $name done ──"
}

run_sweep "Standard"     "config/sweep_standard.yaml"
run_sweep "L1"           "config/sweep_L1.yaml"
run_sweep "L1 unc"       "config/sweep_L1_unc.yaml"
run_sweep "AdamW"        "config/sweep_adamw.yaml"
run_sweep "NGN-MDv1"     "config/sweep_ngnmdv1.yaml"
run_sweep "MuonMax-Momo" "config/sweep_muonmax_momo.yaml"

echo ""
echo "All sweeps complete."
