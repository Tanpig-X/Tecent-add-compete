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
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --batch_size 8 \
    --num_workers 4 \
    --bpr_weight 0.5 \
    --time_attn_bias \
    --add_periodic_time_features \
    --timestamp_tz_offset 28800 \
    --use_sample_time_ns_token \
    --delay_aux_enabled \
    --delay_aux_weight 0.1 \
    --cross_domain_seq_attn \
    "$@"
# Time-feature lessons learned on 200M:
#   --add_periodic_time_features  (A,  sample-level via user_int):    GAINS
#   --use_inter_event_features    (B,  per-token additive):           LOSSES → disabled
#   --use_seq_periodic_time       (A+, per-token additive):           LOSSES → disabled (worse than B)
#   --use_time_ns_token           (C,  NS-token from histograms):     NULL — early epoch +0.008 / late -0.008,
#                                                                     net 0 even after shrunk + dropout.
#                                                                     Signal redundant with time_attn_bias +
#                                                                     per-token time_embedding + A. Disabled.
#   --use_sample_time_ns_token    (NS-form A, dedicated NS token):    to be tested
#
# DIN disabled: 200M observation showed DIN is net-negative even at epoch 1
# (random 1M+1 hash table contaminates NS tokens, and we lose 1 user_ns_token
# slot to make room). Code stays in model.py/train.py for future re-enable
# with a saner config (smaller hash + small init + gate).
#
# --delay_aux_enabled: multi-task auxiliary head. Predicts log1p(label_time -
# timestamp). 100% sample coverage (vs 12.4% positives for CVR) + correlated
# but not identical to main task → forces backbone to encode engagement-
# duration signal as side regularisation.
#
# --cross_domain_seq_attn: each backbone block adds a SHARED cross-attention
# where each Q_i attends to the concatenated seq tokens from ALL 4 domains
# (a/b/c/d). Per-domain specialisation in the existing per-domain cross-attn
# is preserved; the new module layers cross-domain context on top via the
# CrossAttention residual. Pure architectural, 0 data change.
#
# Pattern: NS-token-position adds win, per-token-additive adds lose
# (the d_model channel is already saturated by content + baseline time_bucket).
#
# T constraint with current flags:
#   T = num_queries*num_seq + num_ns
#     = 2*4 + (4 user + 1 user_dense + 2 item + 1 SampleTimeNS) = 16
#   d_model=64 % 16 = 0 ✓
# If you toggle any single NS-emitting flag (SampleTimeNS / TimeNS / DIN),
# bump user_ns_tokens by ±1 to keep T=16 divisible.
