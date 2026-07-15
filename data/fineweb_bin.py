"""Dataloaders for FineWeb shards in the .bin format (256 int32 header, then uint16 GPT-2 tokens).

The token stream is chunked into contiguous (seq_len+1) blocks; the engine slices inputs/targets
out of those. Memmaps are opened lazily so that num_workers > 0 is safe.
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

HEADER_BYTES = 1024  # 256 * int32


def _shard_token_count(path):
  return (os.path.getsize(path) - HEADER_BYTES) // 2  # uint16 -> 2 bytes/token


class FineWebBin(Dataset):
  def __init__(self, shard_paths, seq_len):
    assert len(shard_paths) > 0, 'no shards found'
    self.paths = list(shard_paths)
    self.seq_len = seq_len
    self.block = seq_len + 1
    self.blocks_per_shard = [_shard_token_count(p) // self.block for p in self.paths]
    self.cum = np.cumsum([0] + self.blocks_per_shard)
    self.n = int(self.cum[-1])
    self._mm = [None] * len(self.paths)  # lazily-opened memmaps (per worker)

  def __len__(self):
    return self.n

  def _shard(self, s):
    if self._mm[s] is None:
      self._mm[s] = np.memmap(self.paths[s], dtype=np.uint16, mode='r', offset=HEADER_BYTES)
    return self._mm[s]

  def __getitem__(self, i):
    s = int(np.searchsorted(self.cum, i, side='right') - 1)
    off = (i - int(self.cum[s])) * self.block
    chunk = np.asarray(self._shard(s)[off : off + self.block]).astype(np.int64)
    return {'input_ids': torch.from_numpy(chunk)}


def get_bin_dataloaders(cfg):
  """Train loader over a glob of shards, val loader over one held-out shard.

  Expected cfg keys: train_shards (glob str), val_shard (path), seq_len, micro_batch_size,
  num_workers; optional valid_tokens, sampler_seed.
  """
  train_paths = sorted(glob.glob(cfg.train_shards))
  if not train_paths:
    raise FileNotFoundError(f'no train shards match {cfg.train_shards}')

  train_set = FineWebBin(train_paths, cfg.seq_len)
  gen = torch.Generator().manual_seed(getattr(cfg, 'sampler_seed', None) or 0)
  trainloader = DataLoader(
    train_set,
    sampler=RandomSampler(train_set, generator=gen),
    batch_size=cfg.micro_batch_size,
    num_workers=cfg.num_workers,
    pin_memory=True,
    drop_last=True,
    prefetch_factor=2 if cfg.num_workers > 0 else None,
    persistent_workers=cfg.num_workers > 0,
  )

  val_set = FineWebBin([cfg.val_shard], cfg.seq_len)
  if getattr(cfg, 'valid_tokens', None):
    val_rows = min(len(val_set), cfg.valid_tokens // (cfg.seq_len + 1))
    val_set = torch.utils.data.Subset(val_set, range(val_rows))
  validloader = DataLoader(
    val_set,
    batch_size=cfg.micro_batch_size,
    sampler=SequentialSampler(val_set),
    num_workers=cfg.num_workers,
    pin_memory=True,
    drop_last=True,
    prefetch_factor=2 if cfg.num_workers > 0 else None,
    persistent_workers=False,
  )
  return trainloader, validloader
