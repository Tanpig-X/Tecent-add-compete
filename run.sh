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

# ---- Active config: GroupNSTokenizer driven by ns_groups.json ----
# 40-row smoke-test data → small batch_size + few workers + a handful of epochs.
# AMP / TF32 / cuDNN benchmark are on by default in train.py.
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --batch_size 8 \
#     --num_workers 4 \
#     --buffer_batches 4 \
#     --num_epochs 5 \
#     --eval_every_n_steps 0 \
#     "$@"

# ---- Alternative config: RankMixer NS tokenizer (no ns_groups.json required) ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --num_hyformer_blocks 4\
    --emb_skip_threshold 1000000 \
    --batch_size 256 \
    --num_workers 8 \
    "$@"