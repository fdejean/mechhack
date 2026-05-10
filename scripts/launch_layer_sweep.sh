#!/bin/bash
# Run patch_agent at each of 5 single layers, 20 prompts each.
# Sequential because all share one GPU. Total: ~5 layers * 20 prompts * 90s = ~2.5h.
#
# Prereq: scripts/build_single_layer_directions.py has been run successfully,
# so /scratch/hybrid_v2/gemma4_31b/direction_single_L{16,32,44,48,56}.pt exist.
#
# Usage:
#     export AIAAS_KEY=sk-...
#     nohup bash scripts/launch_layer_sweep.sh > logs/layer_sweep.log 2>&1 &

set -e
cd /scratch/mechhack
mkdir -p logs edit_eval

if [ -z "$AIAAS_KEY" ]; then
    echo "[sweep] AIAAS_KEY is not set. export AIAAS_KEY=sk-... and retry."
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LAYERS="16 32 44 48 56"
EVAL=datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl
LIMIT=20

# Verify direction files exist
for L in $LAYERS; do
    DIR=/scratch/hybrid_v2/gemma4_31b/direction_single_L${L}.pt
    if [ ! -f "$DIR" ]; then
        echo "[sweep] FATAL: missing $DIR"
        echo "[sweep] run: python scripts/build_single_layer_directions.py"
        exit 1
    fi
done

echo "[sweep] starting at $(date)"

for L in $LAYERS; do
    DIR=/scratch/hybrid_v2/gemma4_31b/direction_single_L${L}.pt
    OUT=edit_eval/patch_layer_L${L}.json
    LOG=logs/patch_layer_L${L}.log

    echo "[sweep] === Layer $L === $(date) ==="
    python -u patch_agent.py \
        --eval_set "$EVAL" \
        --direction "$DIR" \
        --model_path /data/Gemma-4-31B-it \
        --output "$OUT" \
        --limit $LIMIT 2>&1 | tee "$LOG"
done

echo "[sweep] all done $(date)"
