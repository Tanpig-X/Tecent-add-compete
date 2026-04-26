#!/bin/bash
# Multi-seed experiment runner.
#   Usage: ./run_exp.sh <name> [extra args passed to train.py]
# Runs 3 seeds, prints per-seed best AUC and the mean.

set -uo pipefail
EXP_NAME="${1:?need exp name}"; shift || true
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXP_DIR="${SCRIPT_DIR}/experiments/runs/${EXP_NAME}"
mkdir -p "${EXP_DIR}"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export TRAIN_DATA_PATH="${SCRIPT_DIR}/data/split_1000"

aucs=()
for SEED in 42 123 2024; do
    RUN_DIR="${EXP_DIR}/seed${SEED}"
    mkdir -p "${RUN_DIR}"
    export TRAIN_CKPT_PATH="${RUN_DIR}/ckpt"
    export TRAIN_LOG_PATH="${RUN_DIR}"
    export TRAIN_TF_EVENTS_PATH="${RUN_DIR}/tb"

    python3 -u "${SCRIPT_DIR}/train.py" \
        --ns_tokenizer_type rankmixer \
        --user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2 \
        --ns_groups_json "" \
        --emb_skip_threshold 1000000 \
        --batch_size 64 --num_workers 4 \
        --seed "${SEED}" \
        "$@" >"${RUN_DIR}/run.out" 2>&1

    BEST=$(grep -oE 'AUC: [0-9.]+' "${RUN_DIR}/train.log" 2>/dev/null \
           | awk '{print $2}' | sort -g | tail -1)
    BEST="${BEST:-NaN}"
    aucs+=("${BEST}")
    echo "  seed=${SEED}: best AUC = ${BEST}"
done

MEAN=$(printf '%s\n' "${aucs[@]}" | awk '{s+=$1; n++} END{if(n>0)printf "%.4f", s/n; else print "NaN"}')
STD=$(printf '%s\n' "${aucs[@]}" | awk -v m="${MEAN}" '{s+=($1-m)*($1-m); n++} END{if(n>0)printf "%.4f", sqrt(s/n); else print "NaN"}')
echo ""
echo "[${EXP_NAME}] mean=${MEAN}  std=${STD}  (seeds: ${aucs[*]})"
echo "${EXP_NAME},${MEAN},${STD},${aucs[*]}" >>"${SCRIPT_DIR}/experiments/results.csv"
