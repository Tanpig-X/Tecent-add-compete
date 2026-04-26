"""Plot the LR schedule used by trainer.py — analytically (no training needed).

Two modes:

1. From CLI args, e.g.
     python3 experiments/plot_lr.py \
         --base_lr 1e-4 --warmup_steps 2000 \
         --lr_decay_steps 200000 --min_lr_factor 0.1 \
         --total_steps 250000 --out lr_schedule.png

2. From a checkpoint's train_config.json:
     python3 experiments/plot_lr.py \
         --config ckpt/global_step5.layer=2.head=4.hidden=64.best_model/train_config.json \
         --total_steps 250000 --out lr_schedule.png

Reads the same lr_lambda formula as trainer.py so the plot matches what
the run will actually do.
"""
import argparse
import json
import math
import os
from typing import Optional

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def lr_curve(
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
    lr_decay_steps: int,
    min_lr_factor: float,
) -> np.ndarray:
    """Replicates trainer.py's lr_lambda × base_lr for steps 0..total_steps-1."""
    steps = np.arange(total_steps, dtype=np.float64)
    lr = np.empty_like(steps)
    for i, step in enumerate(steps):
        if warmup_steps > 0 and step < warmup_steps:
            mult = (step + 1) / warmup_steps
        elif lr_decay_steps <= 0:
            mult = 1.0
        else:
            progress = min(1.0, (step - warmup_steps) / lr_decay_steps)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            mult = min_lr_factor + (1.0 - min_lr_factor) * cos
        lr[i] = base_lr * mult
    return lr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=None,
                   help='train_config.json saved beside a checkpoint; '
                        'reads lr / warmup_steps / lr_decay_steps / min_lr_factor')
    p.add_argument('--base_lr', type=float, default=1e-4)
    p.add_argument('--warmup_steps', type=int, default=2000)
    p.add_argument('--lr_decay_steps', type=int, default=200000)
    p.add_argument('--min_lr_factor', type=float, default=0.1)
    p.add_argument('--total_steps', type=int, default=250000,
                   help='How many steps to plot (extends past lr_decay_steps '
                        'so the flat tail is visible)')
    p.add_argument('--out', type=str, default='lr_schedule.png')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = json.load(f)
        base_lr = float(cfg.get('lr', args.base_lr))
        warmup = int(cfg.get('warmup_steps', args.warmup_steps))
        decay = int(cfg.get('lr_decay_steps', args.lr_decay_steps))
        min_factor = float(cfg.get('min_lr_factor', args.min_lr_factor))
        title_extra = f' (from {os.path.basename(args.config)})'
    else:
        base_lr = args.base_lr
        warmup = args.warmup_steps
        decay = args.lr_decay_steps
        min_factor = args.min_lr_factor
        title_extra = ''

    lr = lr_curve(args.total_steps, base_lr, warmup, decay, min_factor)
    steps = np.arange(args.total_steps)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, lr, lw=1.5)
    if warmup > 0:
        ax.axvline(warmup, color='gray', ls='--', alpha=0.5, label=f'end of warmup ({warmup})')
    if decay > 0:
        ax.axvline(warmup + decay, color='gray', ls=':', alpha=0.5,
                   label=f'end of cosine decay ({warmup + decay})')
    ax.axhline(base_lr * min_factor, color='red', ls=':', alpha=0.4,
               label=f'min lr ({base_lr * min_factor:.2e})')
    ax.set_xlabel('optimizer step')
    ax.set_ylabel('learning rate')
    ax.set_title(f'LR schedule: warmup={warmup} → cosine={decay} → flat'
                 f'{title_extra}')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f'wrote {args.out}')
    print(f'  base_lr={base_lr}, warmup={warmup}, decay={decay}, '
          f'min_factor={min_factor}, total_steps={args.total_steps}')
    print(f'  lr at step 0: {lr[0]:.3e}')
    print(f'  lr at end of warmup (step {warmup}): '
          f'{lr[min(warmup, len(lr)-1)]:.3e}')
    if decay > 0 and warmup + decay < len(lr):
        print(f'  lr at end of decay (step {warmup + decay}): '
              f'{lr[warmup + decay]:.3e}')
    print(f'  lr at last step ({len(lr)-1}): {lr[-1]:.3e}')


if __name__ == '__main__':
    main()
