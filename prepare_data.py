"""Data preparation utility — supports two modes:

1. **Local mode** (no args): extract the bundled ``taac_data_test.zip`` into
   ``./data/taac_data_test/`` and synthesize ``schema.json`` there. Used for
   local smoke tests where the zip travels with the repo.
2. **Remote mode** (``--data_dir <path>``): the data is already on disk
   (delivered by the platform that runs the training job, e.g. via
   ``TRAIN_DATA_PATH``). We only synthesize ``schema.json`` next to the
   existing parquet files. This is what the remote training server needs,
   since the zip is not present there.

Idempotent: re-running over an already-prepared directory is a no-op.
"""

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.types as pat


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(SCRIPT_DIR, 'taac_data_test.zip')
DATA_ROOT = os.path.join(SCRIPT_DIR, 'data')
DEFAULT_DATASET_DIR = os.path.join(DATA_ROOT, 'taac_data_test')


SEQ_PREFIX_RE = re.compile(r'^domain_([a-z])_seq_(\d+)$')
USER_INT_RE = re.compile(r'^user_int_feats_(\d+)$')
ITEM_INT_RE = re.compile(r'^item_int_feats_(\d+)$')
USER_DENSE_RE = re.compile(r'^user_dense_feats_(\d+)$')

VOCAB_MARGIN = 1.2          # vocab_size = ceil(observed_max * margin) + 1
TS_DETECT_THRESHOLD = 1e8   # values larger than this look like a unix timestamp


def _strip_macos_junk(root: str) -> None:
    junk = os.path.join(DATA_ROOT, '__MACOSX')
    if os.path.isdir(junk):
        shutil.rmtree(junk, ignore_errors=True)
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn == '.DS_Store' or fn.startswith('._'):
                try:
                    os.remove(os.path.join(dirpath, fn))
                except OSError:
                    pass


def extract_zip(target_dir: str = DEFAULT_DATASET_DIR) -> None:
    """Extract the bundled zip into ``target_dir`` (local-mode only)."""
    if os.path.isdir(target_dir) and any(
        f.endswith('.parquet') for f in os.listdir(target_dir)
    ):
        print(f'[prepare_data] dataset already present at {target_dir}')
        return
    if not os.path.isfile(ZIP_PATH):
        sys.exit(f'[prepare_data] zip not found: {ZIP_PATH}')
    os.makedirs(DATA_ROOT, exist_ok=True)
    print(f'[prepare_data] extracting {ZIP_PATH} -> {DATA_ROOT}')
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_ROOT)
    _strip_macos_junk(target_dir)


def _vocab_from_int_array(values: np.ndarray) -> int:
    """Derive vocab_size from a 1D numpy int array. Negatives/0 are padding."""
    if values.size == 0:
        return 1
    mx = int(values.max())
    if mx <= 0:
        return 1
    return int(np.ceil(mx * VOCAB_MARGIN)) + 1


def _as_single_array(col) -> pa.Array:
    """Collapse a ChunkedArray (what Table.column returns) into one Array."""
    if isinstance(col, pa.ChunkedArray):
        return col.combine_chunks()
    return col


def _stat_int_column(col) -> Tuple[int, int]:
    """Return (dim, vocab_size) for an int column. dim=1 for scalar, max
    list length for variable-length list columns.
    """
    arr = _as_single_array(col)
    if pat.is_list(arr.type) or pat.is_large_list(arr.type):
        offsets = arr.offsets.to_numpy()
        values = arr.values.fill_null(0).to_numpy(
            zero_copy_only=False).astype(np.int64, copy=False)
        lens = offsets[1:] - offsets[:-1]
        dim = max(int(lens.max()) if lens.size else 1, 1)
        return dim, _vocab_from_int_array(values)
    arr = arr.fill_null(0)
    values = arr.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    return 1, _vocab_from_int_array(values)


def _stat_dense_column(col) -> int:
    """Return dim for a dense column. dim=1 for scalar, else max list length."""
    arr = _as_single_array(col)
    if pat.is_list(arr.type) or pat.is_large_list(arr.type):
        offsets = arr.offsets.to_numpy()
        lens = offsets[1:] - offsets[:-1]
        return max(int(lens.max()) if lens.size else 1, 1)
    return 1


def _is_timestamp_column(col) -> bool:
    """Heuristic: a sequence column is a timestamp iff its max value > 1e8."""
    arr = _as_single_array(col)
    if pat.is_list(arr.type) or pat.is_large_list(arr.type):
        values = arr.values.to_numpy()
        if values.size == 0:
            return False
        return int(values.max()) > TS_DETECT_THRESHOLD
    arr = arr.fill_null(0).to_numpy(zero_copy_only=False)
    if arr.size == 0:
        return False
    return float(np.max(arr)) > TS_DETECT_THRESHOLD


def build_schema(parquet_paths) -> Dict[str, Any]:
    """Scan one or more parquet files and derive the schema.

    When multiple files are passed (e.g. train + valid shards), per-column
    vocab_size and dim are taken as the max across all files so the schema
    covers every observed id.
    """
    if isinstance(parquet_paths, str):
        parquet_paths = [parquet_paths]

    # Read first row group from every file and concatenate per-column.
    column_arrays: Dict[str, List[pa.Array]] = defaultdict(list)
    for p in parquet_paths:
        pf = pq.ParquetFile(p)
        table = pf.read_row_group(0)
        for name in table.column_names:
            column_arrays[name].append(table.column(name).combine_chunks())

    name_to_col: Dict[str, pa.Array] = {}
    for name, arrs in column_arrays.items():
        name_to_col[name] = pa.concat_arrays(arrs) if len(arrs) > 1 else arrs[0]

    user_int: List[List[int]] = []
    item_int: List[List[int]] = []
    user_dense: List[List[int]] = []
    seq_by_domain: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    # value: list of (fid, vocab_size, column_name)

    for name, col in name_to_col.items():
        m = USER_INT_RE.match(name)
        if m:
            fid = int(m.group(1))
            dim, vs = _stat_int_column(col)
            user_int.append([fid, vs, dim])
            continue
        m = ITEM_INT_RE.match(name)
        if m:
            fid = int(m.group(1))
            dim, vs = _stat_int_column(col)
            item_int.append([fid, vs, dim])
            continue
        m = USER_DENSE_RE.match(name)
        if m:
            fid = int(m.group(1))
            dim = _stat_dense_column(col)
            user_dense.append([fid, dim])
            continue
        m = SEQ_PREFIX_RE.match(name)
        if m:
            domain_letter, fid_str = m.group(1), m.group(2)
            domain = f'seq_{domain_letter}'
            fid = int(fid_str)
            _, vs = _stat_int_column(col)
            seq_by_domain[domain].append((fid, vs, name))
            continue
        # Else: meta column (user_id, item_id, label_*, timestamp). Skip.

    # Stable order by fid for deterministic output.
    user_int.sort(key=lambda x: x[0])
    item_int.sort(key=lambda x: x[0])
    user_dense.sort(key=lambda x: x[0])

    seq: Dict[str, Dict[str, Any]] = {}
    for domain, entries in seq_by_domain.items():
        entries.sort(key=lambda x: x[0])
        # ts_fid: first column whose max value exceeds the timestamp threshold.
        ts_fid: Optional[int] = None
        for fid, _, name in entries:
            if _is_timestamp_column(name_to_col[name]):
                ts_fid = fid
                break
        # Sequence prefix is consistent within a domain (we matched on it).
        prefix = f'domain_{domain.split("_")[1]}_seq'
        seq[domain] = {
            'prefix': prefix,
            'ts_fid': ts_fid,
            'features': [[fid, vs] for fid, vs, _ in entries],
        }

    return {
        'user_int': user_int,
        'item_int': item_int,
        'user_dense': user_dense,
        'seq': seq,
    }


def write_schema(
    data_dir: str,
    force: bool = False,
    out_path: Optional[str] = None,
) -> str:
    """Synthesize ``schema.json`` from the parquet files under ``data_dir``.

    By default writes ``<data_dir>/schema.json``. On platforms that mount the
    data directory read-only, pass ``out_path=<some/writable/path>`` and the
    file is written there instead — the scan still happens against
    ``data_dir``. Returns the absolute output path.
    """
    if not os.path.isdir(data_dir):
        sys.exit(f'[prepare_data] data_dir does not exist: {data_dir}')
    schema_path = out_path or os.path.join(data_dir, 'schema.json')
    if os.path.isfile(schema_path) and not force:
        print(f'[prepare_data] schema already exists at {schema_path}, skipping')
        return schema_path

    parquets = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('.parquet')
    )
    if not parquets:
        sys.exit(f'[prepare_data] no .parquet under {data_dir}')

    schema = build_schema(parquets)
    os.makedirs(os.path.dirname(schema_path) or '.', exist_ok=True)
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)

    n_seq_feats = sum(len(d['features']) for d in schema['seq'].values())
    print(
        f'[prepare_data] wrote schema -> {schema_path}\n'
        f'  scanned: {parquets}\n'
        f'  user_int={len(schema["user_int"])} cols, '
        f'item_int={len(schema["item_int"])} cols, '
        f'user_dense={len(schema["user_dense"])} cols, '
        f'seq_domains={len(schema["seq"])} ({n_seq_feats} feats)'
    )
    for domain, cfg in schema['seq'].items():
        print(f'    {domain}: prefix={cfg["prefix"]}, ts_fid={cfg["ts_fid"]}, '
              f'n_feats={len(cfg["features"])}')
    return schema_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='Path to a directory of *.parquet files. When provided, only '
             'synthesize schema.json there (remote mode). When omitted, '
             'extract the local zip into ./data/taac_data_test/ first.'
    )
    parser.add_argument(
        '--force', action='store_true', default=False,
        help='Overwrite an existing schema.json instead of skipping.'
    )
    args = parser.parse_args()

    if args.data_dir:
        write_schema(args.data_dir, force=args.force)
    else:
        extract_zip(DEFAULT_DATASET_DIR)
        write_schema(DEFAULT_DATASET_DIR, force=args.force)


if __name__ == '__main__':
    main()
