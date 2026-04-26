"""Stratified 80/20 split of taac_data/demo_1000.parquet.

Writes two parquet files into data/split_1000/ with the canonical
naming so dataset.get_pcvr_data picks them up via filename split:
  - train_split.parquet
  - valid_split.parquet
"""
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'taac_data', 'demo_1000.parquet')
OUT_DIR = os.path.join(ROOT, 'data', 'split_1000')

VALID_RATIO = 0.20
SEED = 42


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tbl = pq.read_table(SRC)
    n = tbl.num_rows
    label_type = tbl.column('label_type').to_numpy(zero_copy_only=False)
    is_pos = label_type == 2

    rng = np.random.default_rng(SEED)
    pos_idx = np.where(is_pos)[0]
    neg_idx = np.where(~is_pos)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_pos_v = int(round(len(pos_idx) * VALID_RATIO))
    n_neg_v = int(round(len(neg_idx) * VALID_RATIO))
    valid_idx = np.sort(np.concatenate([pos_idx[:n_pos_v], neg_idx[:n_neg_v]]))
    train_idx = np.sort(np.concatenate([pos_idx[n_pos_v:], neg_idx[n_neg_v:]]))

    print(f'src rows={n} (pos={is_pos.sum()}, neg={(~is_pos).sum()})')
    print(f'train rows={len(train_idx)} '
          f'(pos={is_pos[train_idx].sum()}, neg={(~is_pos[train_idx]).sum()})')
    print(f'valid rows={len(valid_idx)} '
          f'(pos={is_pos[valid_idx].sum()}, neg={(~is_pos[valid_idx]).sum()})')

    train_tbl = tbl.take(pa.array(train_idx))
    valid_tbl = tbl.take(pa.array(valid_idx))
    train_path = os.path.join(OUT_DIR, 'train_split.parquet')
    valid_path = os.path.join(OUT_DIR, 'valid_split.parquet')
    pq.write_table(train_tbl, train_path, row_group_size=len(train_idx))
    pq.write_table(valid_tbl, valid_path, row_group_size=len(valid_idx))
    print(f'wrote {train_path}\nwrote {valid_path}')


if __name__ == '__main__':
    main()
