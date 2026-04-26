#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# ---- Data prep ----
# Two modes:
#  1. Remote (recommended on the training platform):
#     TRAIN_DATA_PATH is injected by the platform pointing to a directory
#     of *.parquet files. We don't touch the local zip; if schema.json is
#     missing under TRAIN_DATA_PATH, train.py auto-generates it from the
#     parquet files at startup (see train.py auto-bootstrap branch).
#  2. Local: prefer the user-uploaded ./taac_data/ directory if present
#     (typically demo_1000.parquet + valid_*.parquet); otherwise fall back
#     to extracting the bundled taac_data_test.zip into ./data/taac_data_test/.
if [ -z "${TRAIN_DATA_PATH:-}" ]; then
    if [ -d "${SCRIPT_DIR}/taac_data" ] && \
       ls "${SCRIPT_DIR}/taac_data"/*.parquet >/dev/null 2>&1; then
        export TRAIN_DATA_PATH="${SCRIPT_DIR}/taac_data"
    else
        LOCAL_DATA_DIR="${SCRIPT_DIR}/data/taac_data_test"
        if [ ! -f "${LOCAL_DATA_DIR}/schema.json" ]; then
            python3 "${SCRIPT_DIR}/prepare_data.py"
        fi
        export TRAIN_DATA_PATH="${LOCAL_DATA_DIR}"
    fi
fi

export TRAIN_CKPT_PATH="${TRAIN_CKPT_PATH:-${SCRIPT_DIR}/ckpt}"
export TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-${SCRIPT_DIR}/logs}"
export TRAIN_TF_EVENTS_PATH="${TRAIN_TF_EVENTS_PATH:-${SCRIPT_DIR}/tb}"

# ---------------------------------------------------------------------------
# Pack A config: RankMixer NS tokenizer + RoPE + warmup-cosine LR scheduler.
#
# We use RankMixer (not Group) for the active config because:
#   - GroupNSTokenizer issues ~33 small CUDA ops per forward (11 small cats
#     + 11 small Linears + 11 SiLU), vs RankMixer's ~15. On H20 the
#     per-kernel launch overhead dominates over the actual matmul work,
#     making Group ~3x slower end-to-end at batch_size=256. The semantic
#     prior from official ns_groups doesn't justify a 3x throughput hit at
#     this batch size.
#   - Keep the algorithmic Pack A improvements that DON'T cost throughput:
#     RoPE (small attention overhead), warmup+cosine LR scheduler.
#
# Data-pipeline knobs are kept at baseline-memory values to avoid the OOM
# we hit with batch=1024/buffer=64 earlier.
#
# T = num_queries*num_seq + (user_ns + item_ns + 1 user_dense)
#   = 2*4 + (5 + 2 + 1) = 16
#   d_model=64 % T(16) == 0  ✓
# ---------------------------------------------------------------------------
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --use_rope \
    --rope_base 10000 \
    --batch_size 256 \
    --num_workers 8 \
    --buffer_batches 8 \
    --warmup_steps 2000 \
    --lr_decay_steps 200000 \
    --min_lr_factor 0.1 \
    --lr 1e-4 \
    --sparse_lr 0.05 \
    "$@"

# ---------------------------------------------------------------------------
# Optional: GroupNSTokenizer + official ns_groups.json
#
# Slower (~3x at batch_size=256 due to many small CUDA kernels) but the
# semantic feature prior may give a small AUC lift if you can amortize the
# kernel launches with a much larger batch. Try --batch_size 1024 first;
# if it still OOMs, drop num_workers to 4 and buffer_batches to 4.
# ---------------------------------------------------------------------------
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --use_rope --rope_base 10000 \
#     --batch_size 256 --num_workers 8 --buffer_batches 8 \
#     --warmup_steps 2000 --lr_decay_steps 200000 --min_lr_factor 0.1 \
#     --lr 1e-4 --sparse_lr 0.05 \
#     "$@"

# ---------------------------------------------------------------------------
# After confirming the baseline-memory Pack A runs OK, you can try larger
# batches incrementally. Each ~2x batch roughly doubles the per-worker
# shuffle-buffer footprint; bump conservatively:
#   --batch_size 512 --num_workers 8 --buffer_batches 8       # ~2x baseline mem
#   --batch_size 512 --num_workers 8 --buffer_batches 16      # ~4x baseline mem
#   --batch_size 1024 --num_workers 8 --buffer_batches 8      # ~4x baseline mem
# Longer user history is a separate axis and ALSO doubles per-batch memory:
#   --seq_max_lens 'seq_a:512,seq_b:512,seq_c:1024,seq_d:1024'   # ~2x
# ---------------------------------------------------------------------------
