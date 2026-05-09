# Overnight scripts

Five files. Each is independent.

## What each does

| File | Purpose |
|---|---|
| `run_pre_resumable.py` | Runs `pre_agent` on n=81 with per-prompt checkpointing and restart-on-crash. |
| `build_whitened_direction.py` | Computes the whitened-SVD refusal direction at layer 48 from cached residuals. CPU only. ~5 min. |
| `build_single_layer_directions.py` | Splits the existing 16-layer direction file into 5 single-layer .pt files for the layer sweep. ~5 sec. |
| `launch_layer_sweep.sh` | Loops `patch_agent` over layers 16/32/44/48/56 × 20 prompts each. ~2.5 h. |
| `audit_helper.py` | Pretty-prints judge-flagged compliances in batches for tomorrow morning's manual audit. |

## File placement

All scripts go in `/scratch/mechhack/scripts/`. Make sure `logs/` exists at `/scratch/mechhack/logs/` for nohup output.

```bash
mkdir -p /scratch/mechhack/scripts /scratch/mechhack/logs
# upload the 5 files here
chmod +x /scratch/mechhack/scripts/launch_layer_sweep.sh
```

## Launch sequence (3 pods, in this order)

### Pod 1 — PRE n=81 (the headline number)

```bash
cd /scratch/mechhack
export AIAAS_KEY=sk-zARqZI9EDI7iNADkjcC5iw

# Smoke test on 3 (will resume what's already in the file)
python -u scripts/run_pre_resumable.py \
    --eval_set datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl \
    --direction /scratch/hybrid/gemma4_31b/direction_refusal_gemma4_31b.pt \
    --output edit_eval/pre_n81.json \
    --limit 3 --k 3 2>&1 | tee logs/pre_smoke_wrapper.log

# If smoke OK ([3/3] line printed, no traceback), launch full
nohup python -u scripts/run_pre_resumable.py \
    --eval_set datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl \
    --direction /scratch/hybrid/gemma4_31b/direction_refusal_gemma4_31b.pt \
    --output edit_eval/pre_n81.json \
    --limit 81 --k 3 \
    > logs/pre_n81.log 2>&1 &
echo "PRE PID=$!"
date
```

### Pod 2 — Layer sweep

```bash
cd /scratch/mechhack
export AIAAS_KEY=sk-zARqZI9EDI7iNADkjcC5iw

# Build single-layer direction files (~5 sec)
python scripts/build_single_layer_directions.py

# Smoke at layer 48 with 2 prompts
python -u patch_agent.py \
    --eval_set datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl \
    --direction /scratch/hybrid_v2/gemma4_31b/direction_single_L48.pt \
    --model_path /data/Gemma-4-31B-it \
    --output edit_eval/patch_L48_smoke.json \
    --limit 2 2>&1 | tee logs/patch_L48_smoke.log

# If smoke OK, launch full sweep
nohup bash scripts/launch_layer_sweep.sh > logs/layer_sweep.log 2>&1 &
echo "SWEEP PID=$!"
date
```

### Pod 3 — Whitened SVD ablation

```bash
cd /scratch/mechhack
export AIAAS_KEY=sk-zARqZI9EDI7iNADkjcC5iw

# Step 1: build the whitened direction (CPU, ~5 min)
python scripts/build_whitened_direction.py 2>&1 | tee logs/whitened_build.log

# Step 2: smoke on 2 prompts
python -u patch_agent.py \
    --eval_set datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl \
    --direction /scratch/hybrid_v2/gemma4_31b/direction_arditi_whitened_L48.pt \
    --model_path /data/Gemma-4-31B-it \
    --output edit_eval/patch_whitened_smoke.json \
    --limit 2 2>&1 | tee logs/whitened_smoke.log

# Step 3: full run
nohup python -u patch_agent.py \
    --eval_set datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl \
    --direction /scratch/hybrid_v2/gemma4_31b/direction_arditi_whitened_L48.pt \
    --model_path /data/Gemma-4-31B-it \
    --output edit_eval/patch_whitened.json \
    --limit 20 > logs/patch_whitened.log 2>&1 &
echo "WHITENED PID=$!"
date
```

## Tomorrow morning

```bash
# 1. Check completion
ls -la /scratch/mechhack/edit_eval/pre_n81.json
ls -la /scratch/mechhack/edit_eval/patch_layer_L*.json
ls -la /scratch/mechhack/edit_eval/patch_whitened.json

# 2. Aggregate summaries
for f in pre_n81 patch_whitened; do
  echo "=== $f ==="
  python -c "import json; d=json.load(open('/scratch/mechhack/edit_eval/$f.json')); import json; print(json.dumps(d['aggregate'], indent=2))"
done

for L in 16 32 44 48 56; do
  echo "=== layer L$L ==="
  python -c "import json; d=json.load(open('/scratch/mechhack/edit_eval/patch_layer_L${L}.json')); print(json.dumps(d['aggregate'], indent=2))"
done

# 3. Audit PRE compliances (paste output to chat for review)
cd /scratch/mechhack
python scripts/audit_helper.py --input edit_eval/pre_n81.json --start 0
python scripts/audit_helper.py --input edit_eval/pre_n81.json --start 25
python scripts/audit_helper.py --input edit_eval/pre_n81.json --start 50
```

## If something crashes overnight

Each script auto-resumes if you rerun the same command. The JSON checkpoint after every prompt is the source of truth.

```bash
# Check what's still alive
ps aux | grep -E "pre_agent|patch_agent|run_pre" | grep -v grep

# If pre n=81 died, just relaunch — it'll pick up from where it left off
nohup python -u scripts/run_pre_resumable.py \
    --eval_set datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl \
    --direction /scratch/hybrid/gemma4_31b/direction_refusal_gemma4_31b.pt \
    --output edit_eval/pre_n81.json \
    --limit 81 --k 3 \
    > logs/pre_n81_resume.log 2>&1 &
```

For the layer sweep, if it crashed mid-layer you can edit `LAYERS=` in `launch_layer_sweep.sh` to skip the ones already done, then relaunch.
