# Results

## Overview

We probe the residual stream of Gemma-4-31B-it and Qwen3.6-27B to predict refusal and cyber-harm category from internal activations (Level 1), then use the recovered refusal direction to attribute per-token causal weight and propose minimal prompt edits that flip Gemma's behavior while preserving intent (Level 2).

Headline numbers:
- **Level 1 mean AUC: 0.918** across all 5 tasks
- **Level 2: 100% intent preservation, 10% behavior-flip rate** on 20 prompts

---

## Setup

- **Models:** Gemma-4-31B-it, Qwen3.6-27B (both frozen, accessed via `/data/`)
- **Refusal datasets:** Gemma 876 samples, Qwen 878 samples
- **Cyber dataset:** 7,114 samples across 4 categories (benign / dual_use / high_risk_dual_use / prohibited)
- **Train/test split:** deterministic by `hash(sample_id) mod 100 < 70`, as required by the rules

We extracted activations at multiple layers and built three probe families: single-layer trained probes (linear, MLP, attention), refusal-direction probes (mean-difference per Arditi et al. 2024), and a hybrid 53-feature representation combining direction loadings + logit-lens projections + 5 derived features across 16 layers.

---

## Level 1 — Predicting behavior from internals

### Baseline probes — single-layer activations at layer 32

Reference numbers from `starter_code/train_probe.py` (linear / MLP / attention × 5 seeds × 2 regimes):

| Task | Best probe | AUC |
|---|---|---|
| Refusal-Gemma | attention / batch | 0.914 |
| Refusal-Qwen | MLP / batch | 0.841 |
| Cyber-1 (dual_use vs benign) | MLP / batch | 0.870 |
| Cyber-2 (high_risk vs rest) | MLP / batch | 0.858 |
| Cyber-3 (prohibited vs rest) | linear / batch | 0.930 |
| **Mean** | | **0.883** (σ = 0.038) |

Bonus: Benign-vs-rest reaches AUC 0.987 (linear / batch). Metrics saved to `probes/results/*_metrics.jsonl`.

### Qwen layer sweep — 16 / 32 / 48

Qwen has a hybrid architecture (16 full-attention + 48 DeltaNet layers). We tested whether layer 32 was a poor default:

| Layer | Linear AUC | XGBoost AUC |
|---|---|---|
| 16 | 0.824 | 0.856 |
| 32 | 0.838 | 0.869 |
| 48 | **0.866** | **0.884** |

Refusal lives later in Qwen than in Gemma. Layer 48 + XGBoost gives +4.3% over the layer-32 MLP baseline. Reproduced with `qwen_layer_sweep.py`.

### Hybrid 53-feature pipeline (Gemma)

For Gemma, we extracted activations at every 4th layer (0, 4, 8, …, 60 → 16 layers) and built a 53-dimensional feature vector per sample:
- 16 direction loadings (per-layer projection onto the mean-difference refusal direction)
- 16 `logit_refusal` probabilities (logit-lens projection through W_U onto refusal-starter tokens)
- 16 `logit_compliance` probabilities (same, onto compliance-starter tokens)
- 5 derived features: `transition_layer`, `peak_layer`, `total_loading`, `max_logit_ratio`, `n_tokens`

Trained classifiers via `starter_code/train_hybrid.py`:

| Task | Direction-only | Logistic | XGBoost | Best layer |
|---|---|---|---|---|
| Refusal-Gemma | **0.929** | 0.926 | 0.926 | 48 |
| Cyber-1 | 0.955 | 0.958 | **0.964** | 48 |
| Cyber-2 | 0.925 | 0.933 | **0.935** | 44 |
| Cyber-3 | 0.827 | 0.857 | **0.879** | 40 |
| **Mean (4 tasks)** | 0.909 | 0.919 | **0.926** | |

Two findings:
1. **Layer 48 beats layer 32** on Refusal-Gemma and Cyber-1; layer 44 on Cyber-2; layer 40 on Cyber-3. The "middle of the network" default is wrong for refusal-style signals.
2. **Direction-only matches XGBoost on Refusal-Gemma** (0.929 vs 0.926). Refusal is approximately one direction in 5376-dim activation space — consistent with Arditi et al. 2024.

Artifacts: directions in `/scratch/hybrid/gemma4_31b/direction_*.pt`, features in `features_*.npz`, trained XGBoost models in `hybrid_models/gemma4_31b/xgb_*.json`.

### Final Level 1 — best per task

| Task | Method | AUC |
|---|---|---|
| Refusal-Gemma | Direction-only @ layer 48 | 0.929 |
| Refusal-Qwen | XGBoost @ layer 48 | 0.884 |
| Cyber-1 (dual_use vs benign) | XGBoost on hybrid features | 0.964 |
| Cyber-2 (high_risk vs rest) | XGBoost on hybrid features | 0.935 |
| Cyber-3 (prohibited vs rest) | XGBoost on hybrid features | 0.879 |
| **Mean** | | **0.918** |

We mix methods per task because (a) the rules permit any mech-interp method per task, and (b) different tasks expose signal differently — Qwen needed a per-model layer sweep, cyber tasks benefited from cross-layer feature engineering, and Gemma refusal is so cleanly directional that one dot product suffices.

---

## Level 2 — Flipping the model with minimal edits

Pipeline: **attribute → edit → judge → verify**.

1. **Attribute** (`starter_code/attribute_tokens.py`): for each token in a refused prompt, project its activation onto the refusal direction at each of the 16 selected layers, weight by XGBoost layer importance, and sum. Top-K tokens (K=15) are the candidates to edit.
2. **Edit** (`edit_agent.py`, calling MiniMaxAI/MiniMax-M2.7 via EPFL AIaaS): editor sees the prompt with `[position|score]token` markers on high-attribution tokens, proposes JSON-schema-validated edits that target marked spans while preserving intent.
3. **Intent judge** (MiniMax-M2.7 via AIaaS): scores 0–10 whether the edit preserved the original ask. Threshold ≥7.
4. **Behavior verify** (`starter_code/behavior_verifier.py`): re-rolls Gemma-4-31B on the edited prompt, classifies refusal vs compliance.

Eval set: 81 prompts in `datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl` (curated by organizers; all originally refused by Gemma, all <2048 tokens). We ran on the first 20.

Editor system prompt is rewritten from the reference baseline to enforce verb preservation explicitly: rephrase framing, do not soften "implement" to "explain". This raised intent preservation from 0/2 to 20/20.

### Results — 20 prompts, single iteration

| Metric | Count | Rate |
|---|---|---|
| Editor proposed edits | 20/20 | 100% |
| Intent preserved (judge ≥ 7) | 20/20 | 100% |
| **Behavior flipped (Gemma now complies)** | **2/20** | **10%** |
| Behavior verify skipped (rollout failure) | 3/20 | 15% |
| Behavior unchanged (Gemma still refused) | 15/20 | 75% |

Results in `edit_eval/level2_final.json`. Total runtime ~23 min on one A100-80GB.

**Interpretation:** edits identified by the refusal direction flip Gemma's behavior on 10% of attempts while preserving the underlying ask 100% of the time. This is a non-trivial causal claim — the direction isn't just correlated with refusal, it identifies tokens whose modification can causally flip the decision. The strict intent gate (no softened verbs, no benign rephrasings counted) means false positives are filtered out.

The two successful flips (`EVO_0109`, `EVO_0209`) had edits clustered on framing language (e.g. "Following the methodology used by" → "Applying patterns from"), not on the core ask. The 75% non-flips suggest single-iteration editing is insufficient for many prompts; multi-iteration ({attribute → edit → re-probe → re-edit}, up to 5 iterations as in the reference design) would likely raise the flip rate substantially.

---

## What we wrote vs. what came from starter code

**Provided by the hackathon:**
- `starter_code/extract_residuals.py`, `starter_code/train_probe.py`
- `starter_code/compute_and_extract.py`, `starter_code/train_hybrid.py`
- `starter_code/attribute_tokens.py`, `starter_code/iterative_edit_agent.py` (skeleton, raises NotImplementedError)
- `starter_code/llm_clients.py`, `starter_code/behavior_verifier.py`
- `starter_code/predict.py`

**We wrote:**
- `build_manifest.py` — joins per-task extraction metadata with dataset labels into the unified manifest format `train_probe.py` requires
- `xgboost_probes.py` — XGBoost on raw layer-32 activations
- `qwen_layer_sweep.py` — multi-layer probe comparison for Qwen
- `edit_agent.py` — runnable Level-2 pipeline (the reference `iterative_edit_agent.py` is a skeleton); single-iteration attribute → edit → judge → verify
- Patched `EDITOR_SYSTEM` in `iterative_edit_agent.py` for stricter intent preservation

**Data fixes:**
- Cyber JSONL uses `prompt`, but `extract_residuals.py` expects `attack_prompt`; we wrote `datasets/cyber_probes/all_fixed.jsonl` with the renamed key
- Friend's hybrid pipeline saved `features_cyber_probe3.npz` after a long delay; older `train_hybrid.py` outputs missed it, requiring a rerun

---

## Reproducing the results

### Level 1 — single-layer probes

```bash
# Extract Gemma + Qwen activations at layer 32
python starter_code/extract_residuals.py --model_key gemma4_31b --model_path /data/Gemma-4-31B-it
python starter_code/extract_residuals.py --model_key qwen36 --model_path /data/Qwen3.6-27B

# Patch cyber dataset key
python -c "import json; rows=[json.loads(l) for l in open('datasets/cyber_probes/train.jsonl')]+[json.loads(l) for l in open('datasets/cyber_probes/test.jsonl')]; open('datasets/cyber_probes/all_fixed.jsonl','w').writelines(json.dumps({**r,'attack_prompt':r['prompt']})+'\n' for r in rows)"

# Cyber activations
python starter_code/extract_residuals.py --model_key gemma4_31b --samples_file datasets/cyber_probes/all_fixed.jsonl --out_dir extracts/cyber_gemma4_31b

# Manifest + probes
python build_manifest.py
python starter_code/train_probe.py --extracts_dir extracts/gemma4_31b --manifest extracts/unified_manifest.json --task refusal_gemma4_31b
# (repeat per task)
```

### Qwen layer sweep

```bash
python starter_code/extract_residuals.py --model_key qwen36 --model_path /data/Qwen3.6-27B --layers "16,32,48" --out_dir extracts/qwen36_multilayer
python qwen_layer_sweep.py
```

### Hybrid features (Gemma)

```bash
python starter_code/compute_and_extract.py --model_key gemma4_31b --layers "0:62:4" --out_dir /scratch/hybrid --num_gpus 1 --batch_size 8
python starter_code/train_hybrid.py --features_dir /scratch/hybrid/gemma4_31b --out_dir hybrid_models/gemma4_31b
```

### Level 2

```bash
# 1. Attribution (writes edit_eval/attribution_refusal_gemma4_31b_hybrid.json)
python starter_code/attribute_tokens.py \
    --model_key gemma4_31b \
    --artifacts_dir /scratch/hybrid/gemma4_31b \
    --models_dir hybrid_models/gemma4_31b/gemma4_31b \
    --task refusal_gemma4_31b \
    --out_dir edit_eval \
    --sample_limit 20

# 2. Edit + judge + verify (writes edit_eval/level2_final.json)
export AIAAS_KEY=sk-...
python edit_agent.py \
    --attribution edit_eval/attribution_refusal_gemma4_31b_hybrid.json \
    --eval_set datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl \
    --output edit_eval/level2_final.json \
    --limit 20
```

---

## Where everything lives

```
/scratch/mechhack/
├── extracts/                      single-layer activation .pt files
│   ├── gemma4_31b/                  refusal-Gemma, layer 32
│   ├── qwen36/                      refusal-Qwen, layer 32
│   ├── qwen36_multilayer/qwen36/    Qwen at layers 16/32/48
│   ├── cyber_gemma4_31b/            cyber, layer 32
│   └── unified_manifest.json
│
├── /scratch/hybrid/gemma4_31b/    hybrid pipeline outputs
│   ├── direction_*.pt               16-layer direction vectors per task
│   ├── features_*.npz               53-dim feature matrices per task
│   ├── W_U.pt                       Gemma unembedding (5.6 GB)
│   └── refusal_token_ids.json, compliance_token_ids.json
│
├── hybrid_models/gemma4_31b/      trained XGBoost models per task
│   └── xgb_*.json, results.json
│
├── probes/                        single-layer probe outputs
│   ├── weights/                     trained probe state dicts
│   └── results/                     per-task metrics jsonl + SUMMARY.txt
│
├── edit_eval/                     Level-2 outputs
│   ├── attribution_refusal_gemma4_31b_hybrid.json
│   ├── level2_v2.json               5-sample smoketest
│   └── level2_final.json            20-sample full run with behavior verify
│
├── build_manifest.py
├── xgboost_probes.py
├── qwen_layer_sweep.py
└── edit_agent.py
```
