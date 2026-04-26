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
# Pack A config: GroupNSTokenizer (official ns_groups.json) + RoPE +
#   warmup-cosine LR scheduler + bigger batch / longer user history.
#
# Defaults below are tuned for 200M-scale Taiji runs on H20 (97 GB).
# For the local 1000-row smoke test you can override with smaller values:
#     bash run.sh --batch_size 64 --warmup_steps 50 --lr_decay_steps 0
#
# T = num_queries*num_seq_domains + num_ns
#   = 1*4 + (7 user_int + 1 user_dense + 4 item_int) = 16
#   d_model=64 % T(16) == 0  ✓
# ---------------------------------------------------------------------------
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type group \
    --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
    --num_queries 1 \
    --emb_skip_threshold 1000000 \
    --use_rope \
    --rope_base 10000 \
    --batch_size 1024 \
    --num_workers 16 \
    --buffer_batches 64 \
    --seq_max_lens 'seq_a:512,seq_b:512,seq_c:1024,seq_d:1024' \
    --warmup_steps 2000 \
    --lr_decay_steps 200000 \
    --min_lr_factor 0.1 \
    --lr 2e-4 \
    --sparse_lr 0.05 \
    "$@"

# ---------------------------------------------------------------------------
# Backup config: RankMixer NS tokenizer (no ns_groups.json needed)
# Use this if the official ns_groups becomes unavailable.
# ---------------------------------------------------------------------------
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type rankmixer \
#     --user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2 \
#     --ns_groups_json "" \
#     --emb_skip_threshold 1000000 \
#     --use_rope --rope_base 10000 \
#     --batch_size 1024 --num_workers 16 --buffer_batches 64 \
#     --seq_max_lens 'seq_a:512,seq_b:512,seq_c:1024,seq_d:1024' \
#     --warmup_steps 2000 --lr_decay_steps 200000 --min_lr_factor 0.1 \
#     --lr 2e-4 --sparse_lr 0.05 \
#     "$@"
