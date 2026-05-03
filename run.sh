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
# --bpr_weight 0.5: in-batch pairwise BPR loss added on top of BCE. Directly
#                  optimises AUC's pairwise ordering metric. Set to 0 to disable.
# --time_attn_bias: TIN-style time-aware attention bias on backbone self-attn
#                  + cross-attn. Per-head learnable alpha (init=0, equivalent to
#                  baseline at start). Forces recency-favouring on cross-attn
#                  and pairwise-time-distance penalty on seq self-attn.
# --din_enabled:   DIN target attention. Hash-embeds item_id (default 1M
#                  buckets) and uses the same emb for target item AND user
#                  item-id history (seq_c[fid=47]). Output added as 1 NS token.
#                  Critical because that history's vocab is ~3.34e8 — too big
#                  for a regular embedding (otherwise zeroed by emb_skip).
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 2 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --batch_size 8 \
    --num_workers 4 \
    --bpr_weight 0.5 \
    --time_attn_bias \
    --din_enabled \
    --din_hash_size 1000000 \
    --din_history_domain seq_c \
    --din_history_fid 47 \
    --add_periodic_time_features \
    --timestamp_tz_offset 28800 \
    --use_time_ns_token \
    --use_sample_time_ns_token \
    "$@"
# Time-feature lessons learned on 200M:
#   --add_periodic_time_features  (A,  sample-level via user_int):    GAINS
#   --use_inter_event_features    (B,  per-token additive):           LOSSES → disabled
#   --use_seq_periodic_time       (A+, per-token additive):           LOSSES → disabled (worse than B)
#   --use_time_ns_token           (C,  NS-token from histograms):     GAINS
#   --use_sample_time_ns_token    (NS-form A, dedicated NS token):    to be tested
# Pattern: NS-token-position adds win, per-token-additive adds lose
# (the d_model channel is already saturated by content + baseline time_bucket).
#
# T constraint with current flags:
#   T = num_queries*num_seq + num_ns
#     = 2*4 + (2 user + 1 user_dense + 2 item + 1 DIN + 1 TimeNS + 1 SampleTimeNS) = 16
#   d_model=64 % 16 = 0 ✓
# If you toggle any single NS-emitting flag (DIN / TimeNS / SampleTimeNS),
# bump user_ns_tokens by ±1 to keep T=16 divisible.
