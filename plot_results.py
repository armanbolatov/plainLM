"""Render the averaging-vs-decay figure from an eval_averages.py json."""

import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

Y_MAX = 5.0
TICKS = [3.3, 3.4, 3.5, 3.75, 4.0, 4.5, 5.0]


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('results')
  p.add_argument('out')
  p.add_argument('--title', default='')
  return p.parse_args()


def main():
  args = parse_args()
  with open(args.results) as f:
    res = json.load(f)
  tokens = lambda step: step * res['tokens_per_step'] / 1e6

  curves, wsd, wsm = res['curves'], res['wsd'], res['wsm']
  grid = sorted(int(k) for k in curves)
  x = [tokens(h) for h in grid]
  col = lambda key: [curves[str(h)][key] for h in grid]
  best_mu = [min(curves[str(h)][f'mu_{mu:g}'] for mu in res['mus']) for h in grid]

  plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
  fig, ax = plt.subplots(figsize=(9, 5.6))
  ax.plot(x, col('const_last'), color='#8a8a8a', lw=1.6, label='const LR + last')
  ax.plot(x, col('uniform'), color='#ff7f0e', lw=1.8, label='uniform (Polyak, from start)')
  ax.plot(x, col('gw_2'), color='#bcbd22', lw=1.8, label=r'$\theta=2/t$ ($\approx$ linear weights)')
  ax.plot(x, col('gw_16'), color='#17becf', lw=2, label=r'$\theta=16/t$ (poly weights)')
  ax.plot(x, best_mu, color='#1f77b4', lw=2.4, label=r'matched $\theta=\mu\eta$, best $\mu$')

  horizons = sorted(int(k) for k in wsd)
  hx = [tokens(h) for h in horizons]
  ax.plot(hx, [wsd[str(h)]['decay_last'] for h in horizons], 's', color='#d62728', ms=8,
          label='WSD decay + last', zorder=5)
  ax.plot(hx, [min(v for k, v in wsd[str(h)].items() if k.startswith('ema_')) for h in horizons],
          '*', color='#e377c2', ms=13, label=r'fixed-$\alpha$ EMA on WSD run', zorder=6)
  for curve, style, color, label in [('ema', '--o', '#2ca02c', 'WSM merge, EMA curve'),
                                     ('sqrt', '--v', '#9467bd', r'WSM merge, $1-\sqrt{t}$ curve')]:
    y = [min(v for k, v in wsm[str(h)].items() if k.startswith(curve + '_')) for h in horizons]
    ax.plot(hx, y, style, color=color, lw=1.5, ms=5, alpha=.85, label=label, zorder=4)

  ax.set_xlabel('training horizon  (million tokens)')
  ax.set_ylabel('validation loss')
  ax.set_xlim(left=0)
  ax.set_yscale('log')
  y_min = min(wsd[str(h)]['decay_last'] for h in horizons) * 0.995
  ax.set_ylim(y_min, Y_MAX)
  ax.yaxis.set_major_locator(FixedLocator([t for t in TICKS if y_min <= t <= Y_MAX]))
  ax.yaxis.set_minor_locator(NullLocator())
  formatter = ScalarFormatter()
  formatter.set_scientific(False)
  ax.yaxis.set_major_formatter(formatter)
  ax.set_title(args.title)
  ax.grid(True, alpha=.25)
  ax.legend(frameon=False, fontsize=9)
  fig.tight_layout()
  fig.savefig(args.out, dpi=140)
  print(f'saved {args.out}')


if __name__ == '__main__':
  main()
