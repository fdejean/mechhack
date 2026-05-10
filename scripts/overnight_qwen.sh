#!/bin/bash
# Overnight chain for qwen-pod.
# Assumes build_whitened_direction.py is currently running.
# This script:
#   1. WAITS for the whitened direction file to exist
#   2. Smokes whitened ablation on n=2
#   3. Runs whitened ablation full n=20
#   4. Runs PRE n=81 using whitened direction (does cleaner direction guide better attacks?)
#   5. Runs PRE on cyber-3 prohibited probe direction (cyber cascade flip stretch goal)
#   6. Runs multilayer mean-pool ablation n=20 (the file we built earlier but never tested)
#
# Failure of any single step does NOT stop the chain.
#
# Usage:
#   nohup bash scripts/overnight_qwen.sh > logs/overnight_qwen.log 2>&1 &

cd /scratch/mechhack
export AIAAS_KEY=${AIAAS_KEY:?AIAAS_KEY must be set}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs edit_eval

EVAL=datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl
WHITENED_DIR=/scratch/hybrid_v2/gemma4_31b/direction_arditi_whitened_L48.pt
MULTILAYER_DIR=/scratch/hybrid_v2/gemma4_31b/direction_arditi_meanpool_multilayer.pt
CYBER3_DIR=/scratch/hybrid/gemma4_31b/direction_cyber_probe3.pt

echo "[qwen-master] $(date) starting"

# -----------------------------------------------------------------------------
# STEP 1: wait for whitened direction file
# -----------------------------------------------------------------------------
echo "[qwen-master] $(date) waiting for whitened direction file..."
WAIT=0
while [ ! -f "$WHITENED_DIR" ]; do
    sleep 30
    WAIT=$((WAIT + 30))
    if [ $WAIT -gt 1800 ]; then
        echo "[qwen-master] $(date) gave up waiting for $WHITENED_DIR after 30 min"
        break
    fi
done
echo "[qwen-master] $(date) whitened file ready (or timed out, will check)"

# -----------------------------------------------------------------------------
# STEP 2: smoke whitened patch_agent (n=2)
# -----------------------------------------------------------------------------
if [ -f "$WHITENED_DIR" ]; then
    echo "[qwen-master] $(date) === whitened smoke (n=2) ==="
    python -u patch_agent.py \
        --eval_set "$EVAL" \
        --direction "$WHITENED_DIR" \
        --model_path /data/Gemma-4-31B-it \
        --output edit_eval/patch_whitened_smoke.json \
        --limit 2 > logs/patch_whitened_smoke.log 2>&1 \
        || echo "[qwen-master] whitened smoke failed but continuing"

    # -----------------------------------------------------------------------------
    # STEP 3: whitened patch_agent full (n=20)
    # -----------------------------------------------------------------------------
    echo "[qwen-master] $(date) === whitened full (n=20) ==="
    python -u patch_agent.py \
        --eval_set "$EVAL" \
        --direction "$WHITENED_DIR" \
        --model_path /data/Gemma-4-31B-it \
        --output edit_eval/patch_whitened_n20.json \
        --limit 20 > logs/patch_whitened_n20.log 2>&1 \
        || echo "[qwen-master] whitened full failed but continuing"

    # -----------------------------------------------------------------------------
    # STEP 4: PRE n=81 using whitened direction (does cleaner direction guide better edits?)
    # -----------------------------------------------------------------------------
    echo "[qwen-master] $(date) === PRE n=81 with whitened direction ==="
    python -u scripts/run_pre_resumable.py \
        --eval_set "$EVAL" \
        --direction "$WHITENED_DIR" \
        --output edit_eval/pre_whitened_n81.json \
        --limit 81 --k 3 > logs/pre_whitened_n81.log 2>&1 \
        || echo "[qwen-master] PRE-whitened failed but continuing"
else
    echo "[qwen-master] $(date) whitened file does not exist, skipping steps 2-4"
fi

# -----------------------------------------------------------------------------
# STEP 5: cyber-cascade flip stretch goal
# -----------------------------------------------------------------------------
# IMPORTANT: This runs PRE with the cyber-3 (prohibited) probe direction as the
# attribution signal. The behavior judge in pre_agent classifies refusal vs
# compliance, which is NOT the same as "is this prompt now classified as benign
# by the cyber probe stack". Treat this as: "did Gemma comply with a prompt
# that was originally cyber-prohibited, AFTER PRE rewriting guided by the
# probe-3 direction?"
#
# We use the cyber eval set, not the refusal eval set, so the prompts are
# legitimately cyber-classified.
# -----------------------------------------------------------------------------
if [ -f "$CYBER3_DIR" ]; then
    echo "[qwen-master] $(date) === cyber-cascade flip stretch goal ==="
    # Build a small eval set of cyber-3 (prohibited) prompts
    python -c "
import json
src = 'datasets/cyber_probes/all_fixed.jsonl'
out = 'datasets/cyber_probes/cyber3_eval_subset.jsonl'
n_kept = 0
with open(src) as f, open(out, 'w') as g:
    for line in f:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('category') == 'prohibited' and r.get('split') == 'test':
            # Map fields to what pre_agent expects
            prompt = r.get('attack_prompt') or r.get('prompt')
            if not prompt or len(prompt) > 8000:
                continue
            entry = {
                'sample_id': r.get('sample_id', f'cyber3_{n_kept}'),
                'attack_prompt': prompt,
                'category': r.get('category'),
                'split': r.get('split'),
            }
            g.write(json.dumps(entry) + '\n')
            n_kept += 1
            if n_kept >= 30:
                break
print(f'wrote {n_kept} cyber-3 prompts to {out}')
"

    python -u scripts/run_pre_resumable.py \
        --eval_set datasets/cyber_probes/cyber3_eval_subset.jsonl \
        --direction "$CYBER3_DIR" \
        --output edit_eval/pre_cyber3_cascade.json \
        --limit 30 --k 3 > logs/pre_cyber3_cascade.log 2>&1 \
        || echo "[qwen-master] cyber cascade failed but continuing"
else
    echo "[qwen-master] $(date) $CYBER3_DIR not found, skipping cyber cascade"
fi

# -----------------------------------------------------------------------------
# STEP 6: multilayer mean-pool ablation
# -----------------------------------------------------------------------------
if [ -f "$MULTILAYER_DIR" ]; then
    echo "[qwen-master] $(date) === multilayer mean-pool ablation (n=20) ==="
    python -u patch_agent.py \
        --eval_set "$EVAL" \
        --direction "$MULTILAYER_DIR" \
        --model_path /data/Gemma-4-31B-it \
        --output edit_eval/patch_multilayer_meanpool_n20.json \
        --limit 20 > logs/patch_multilayer_meanpool_n20.log 2>&1 \
        || echo "[qwen-master] multilayer mean-pool failed but continuing"
else
    echo "[qwen-master] $(date) $MULTILAYER_DIR not found, skipping multilayer"
fi

echo "[qwen-master] $(date) ALL DONE"

# -----------------------------------------------------------------------------
# Print summaries for tomorrow morning
# -----------------------------------------------------------------------------
echo ""
echo "=== QWEN-POD SUMMARY ==="
for f in patch_whitened_n20 pre_whitened_n81 pre_cyber3_cascade patch_multilayer_meanpool_n20; do
    echo ""
    echo "--- $f ---"
    python -c "
import json
try:
    d = json.load(open('edit_eval/${f}.json'))
    print(json.dumps(d.get('aggregate', {}), indent=2))
except Exception as e:
    print(f'(error: {e})')
" || true
done
