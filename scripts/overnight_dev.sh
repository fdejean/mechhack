#!/bin/bash
# Overnight chain for dev-pod.
# Assumes the K=3 PRE n=81 job is already running (started earlier).
# This script:
#   1. WAITS for the K=3 job to finish (polls every 60s)
#   2. Runs PRE n=81 with K=7
#   3. Runs PRE n=81 with K=10
#
# Failure of any single step does NOT stop the chain.
#
# Usage:
#   nohup bash scripts/overnight_dev.sh > logs/overnight_dev.log 2>&1 &

cd /scratch/mechhack
export AIAAS_KEY=${AIAAS_KEY:-***REDACTED-AIAAS-KEY***}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs edit_eval

EVAL=datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl
DIR=/scratch/hybrid/gemma4_31b/direction_refusal_gemma4_31b.pt

echo "[dev-master] $(date) starting"

# -----------------------------------------------------------------------------
# STEP 1: wait for the existing K=3 job to finish
# -----------------------------------------------------------------------------
echo "[dev-master] $(date) waiting for run_pre_resumable (K=3) to finish..."
while pgrep -f "run_pre_resumable.py.*pre_n81.json" > /dev/null; do
    sleep 60
done
echo "[dev-master] $(date) K=3 job finished, summary:"
python -c "
import json
try:
    d = json.load(open('edit_eval/pre_n81.json'))
    print(json.dumps(d.get('aggregate', {}), indent=2))
except Exception as e:
    print(f'(could not read summary: {e})')
" || true

# -----------------------------------------------------------------------------
# STEP 2: PRE K=7 on n=81
# -----------------------------------------------------------------------------
echo "[dev-master] $(date) === PRE K=7 n=81 ==="
python -u scripts/run_pre_resumable.py \
    --eval_set "$EVAL" \
    --direction "$DIR" \
    --output edit_eval/pre_k7_n81.json \
    --limit 81 --k 7 \
    > logs/pre_k7_n81.log 2>&1 \
    || echo "[dev-master] K=7 run failed but continuing"

echo "[dev-master] $(date) K=7 finished, summary:"
python -c "
import json
try:
    d = json.load(open('edit_eval/pre_k7_n81.json'))
    print(json.dumps(d.get('aggregate', {}), indent=2))
except Exception as e:
    print(f'(could not read summary: {e})')
" || true

# -----------------------------------------------------------------------------
# STEP 3: PRE K=10 on n=81
# -----------------------------------------------------------------------------
echo "[dev-master] $(date) === PRE K=10 n=81 ==="
python -u scripts/run_pre_resumable.py \
    --eval_set "$EVAL" \
    --direction "$DIR" \
    --output edit_eval/pre_k10_n81.json \
    --limit 81 --k 10 \
    > logs/pre_k10_n81.log 2>&1 \
    || echo "[dev-master] K=10 run failed but continuing"

echo "[dev-master] $(date) K=10 finished, summary:"
python -c "
import json
try:
    d = json.load(open('edit_eval/pre_k10_n81.json'))
    print(json.dumps(d.get('aggregate', {}), indent=2))
except Exception as e:
    print(f'(could not read summary: {e})')
" || true

echo "[dev-master] $(date) ALL DONE"
