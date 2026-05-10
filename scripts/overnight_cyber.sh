#!/bin/bash
# Overnight chain for cyber-pod.
# Assumes the 5-layer sweep (launch_layer_sweep.sh, layers 16/32/44/48/56)
# is already running.
# This script:
#   1. WAITS for the current sweep to finish (polls every 60s)
#   2. Runs 9 additional layers at every-4-layer granularity:
#        4, 8, 12, 20, 24, 28, 36, 40, 52
#      (gives full curve when combined with the original 5)
#
# Failure of any single layer does NOT stop the chain.
#
# Usage:
#   nohup bash scripts/overnight_cyber.sh > logs/overnight_cyber.log 2>&1 &

cd /scratch/mechhack
export AIAAS_KEY=${AIAAS_KEY:?AIAAS_KEY must be set}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs edit_eval

EVAL=datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl
LIMIT=20
EXTRA_LAYERS="4 8 12 20 24 28 36 40 52"

echo "[cyber-master] $(date) starting"

# -----------------------------------------------------------------------------
# STEP 1: wait for the current layer sweep (any patch_agent on layer-sweep) to finish
# -----------------------------------------------------------------------------
echo "[cyber-master] $(date) waiting for current sweep to finish..."
while pgrep -f "patch_agent.py.*direction_single_L" > /dev/null \
   || pgrep -f "launch_layer_sweep.sh" > /dev/null; do
    sleep 60
done
echo "[cyber-master] $(date) initial sweep finished"

# -----------------------------------------------------------------------------
# STEP 2: build single-layer direction files for the new layers
# (some may already exist from the original split; this is idempotent)
# -----------------------------------------------------------------------------
echo "[cyber-master] $(date) building extra single-layer direction files..."
python -c "
import torch
from pathlib import Path

SRC = '/scratch/hybrid/gemma4_31b/direction_refusal_gemma4_31b.pt'
OUT_DIR = Path('/scratch/hybrid_v2/gemma4_31b')
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXTRA = [4, 8, 12, 20, 24, 28, 36, 40, 52]

src = torch.load(SRC, weights_only=False, map_location='cpu')
all_layers = src['layer_idxs']
print(f'[split-extra] source has layers {all_layers}')

for L in EXTRA:
    if L not in all_layers:
        print(f'[split-extra] WARN: layer {L} not in source, skipping')
        continue
    idx = all_layers.index(L)
    direction_at_L = src['direction'][idx]
    norm_at_L = src['norms'][idx]
    raw = direction_at_L.norm().item()
    if raw < 0.5:
        print(f'[split-extra] layer {L} degenerate (norm={raw:.4f}), skipping')
        continue
    norm_val = norm_at_L.item() if hasattr(norm_at_L, 'item') else float(norm_at_L)
    out = {
        'direction': direction_at_L.unsqueeze(0).float(),
        'norms': torch.tensor([norm_val]).float(),
        'layer_idxs': [L],
        'task': f'refusal_single_L{L}',
    }
    path = OUT_DIR / f'direction_single_L{L}.pt'
    torch.save(out, path)
    print(f'[split-extra] saved L{L} -> {path}')
"

# -----------------------------------------------------------------------------
# STEP 3: sweep each extra layer
# -----------------------------------------------------------------------------
for L in $EXTRA_LAYERS; do
    DIR=/scratch/hybrid_v2/gemma4_31b/direction_single_L${L}.pt
    OUT=edit_eval/patch_layer_L${L}.json
    LOG=logs/patch_layer_L${L}.log

    if [ ! -f "$DIR" ]; then
        echo "[cyber-master] $(date) L${L}: missing $DIR, skipping"
        continue
    fi

    echo "[cyber-master] $(date) === Layer $L ==="
    python -u patch_agent.py \
        --eval_set "$EVAL" \
        --direction "$DIR" \
        --model_path /data/Gemma-4-31B-it \
        --output "$OUT" \
        --limit $LIMIT > "$LOG" 2>&1 \
        || echo "[cyber-master] L${L} failed but continuing"
done

echo "[cyber-master] $(date) ALL DONE"

# -----------------------------------------------------------------------------
# STEP 4: print summary table for tomorrow morning
# -----------------------------------------------------------------------------
echo ""
echo "=== LAYER SWEEP SUMMARY ==="
python -c "
import json, glob
files = sorted(glob.glob('edit_eval/patch_layer_L*.json'))
print(f'{\"layer\":>6} | {\"baseline\":>8} | {\"ablated\":>7} | {\"random\":>6}')
print('-' * 40)
def L_of(f): return int(f.split('_L')[-1].split('.')[0])
for f in sorted(files, key=L_of):
    try:
        d = json.load(open(f))
        agg = d.get('aggregate', {})
        L = L_of(f)
        b = agg.get('baseline_compliance', '?')
        a = agg.get('ablated_compliance', '?')
        r = agg.get('random_compliance', '?')
        print(f'L{L:>4} | {b:>8} | {a:>7} | {r:>6}')
    except Exception as e:
        print(f'{f}: error {e}')
"
