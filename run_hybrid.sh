#!/usr/bin/env bash
# =============================================================
# Hybrid Pipeline — RunAI Launch Script
# =============================================================
# Replace gNN with your actual group number (e.g., g03)
#
# Usage:
#   1. Edit GROUP_NUM below
#   2. chmod +x run_hybrid.sh
#   3. ./run_hybrid.sh
# =============================================================

set -euo pipefail

# ──────────────────────────────────────────────────
# CONFIG — EDIT THIS
# ──────────────────────────────────────────────────
GROUP_NUM="gNN"   # ← CHANGE THIS to your group (e.g., g03)
POD_NAME="hybrid-pipeline"

IMAGE="registry.rcp.epfl.ch/mlo-protsenk/redteam-mechinterp:v9"
PVC_SCRATCH="hackathon-mechhack-scratch-${GROUP_NUM}:/scratch"
PVC_DATA="hackathon-mechhack-shared-ro:/data"

# ──────────────────────────────────────────────────
# STEP 1: Submit the pod
# ──────────────────────────────────────────────────
echo ">>> Submitting pod: ${POD_NAME}"
runai submit "${POD_NAME}" \
    --image "${IMAGE}" \
    --gpu 1 \
    --pvc "${PVC_SCRATCH}" \
    --pvc "${PVC_DATA}" \
    --command -- sleep infinity

echo ">>> Waiting for pod to be ready..."
sleep 10

# ──────────────────────────────────────────────────
# STEP 2: Copy your code to the pod's /scratch
# ──────────────────────────────────────────────────
echo ">>> Pod should be starting. Once ready, exec in with:"
echo ""
echo "    runai exec -it ${POD_NAME} -- bash"
echo ""
echo ">>> Then run these commands inside the pod:"

cat << 'COMMANDS'

# =============================================================
# INSIDE THE POD — paste these commands
# =============================================================

# 1. Clone or copy your repo to /scratch
cd /scratch
git clone <YOUR_REPO_URL> mechhack || true
cd mechhack

# 2. Quick smoke test (2 samples, ~2 min)
echo "=== Smoke test ==="
python starter_code/compute_and_extract.py \
    --model_key gemma4_31b \
    --layers "0:62:4" \
    --out_dir /scratch/hybrid_test \
    --sample_limit 2

# 3. Full extraction — Gemma (~2-3 hours)
echo "=== Full Gemma extraction ==="
python starter_code/compute_and_extract.py \
    --model_key gemma4_31b \
    --layers "0:62:4" \
    --out_dir /scratch/hybrid

# 4. Train classifiers (CPU, ~10 min)
echo "=== Training ==="
python starter_code/train_hybrid.py \
    --features_dir /scratch/hybrid/gemma4_31b \
    --out_dir /scratch/hybrid_models \
    --model_key gemma4_31b

# 5. Check results
cat /scratch/hybrid_models/gemma4_31b/results.json

# 6. (Optional) Qwen extraction (~2-3 hours)
python starter_code/compute_and_extract.py \
    --model_key qwen36 \
    --layers "0:66:4" \
    --out_dir /scratch/hybrid

python starter_code/train_hybrid.py \
    --features_dir /scratch/hybrid/qwen36 \
    --out_dir /scratch/hybrid_models \
    --model_key qwen36

# 7. (Stretch) Level 2 — Token attribution for refusal flipping
python starter_code/attribute_tokens.py \
    --model_key gemma4_31b \
    --artifacts_dir /scratch/hybrid/gemma4_31b \
    --models_dir /scratch/hybrid_models/gemma4_31b \
    --out_dir /scratch/edit_eval \
    --sample_limit 20

COMMANDS

echo ""
echo ">>> Feature importance plots will be in: /scratch/hybrid_models/gemma4_31b/"
echo ">>> Layer AUC curves will be in:         /scratch/hybrid_models/gemma4_31b/"
echo ">>> Results JSON will be in:             /scratch/hybrid_models/gemma4_31b/results.json"
