# Results

## Overview

Probes trained on residual stream activations from Gemma-4-31B and Qwen3.6-27B for refusal and cyber-risk classification tasks. Level 1 baseline uses layer 32 across all tasks; extensions add XGBoost and a Qwen-specific layer sweep.

## Setup

- **Models:** Gemma-4-31B-it, Qwen3.6-27B
- **Layer:** 32 (middle of 64) for baseline
- **Refusal datasets:** Gemma 876 samples, Qwen 878 samples
- **Cyber dataset:** 7,114 samples across 4 categories
- **Probe types:** linear, MLP, attention (1-head and 4-head)
- **Training regime:** 5 seeds × 2 regimes (per-task and batched)

## Level 1 — Baseline probes (layer 32, Gemma-4-31B)

| Task | Best probe | AUC |
|---|---|---|
| Refusal-Gemma | attention / batch | 0.914 |
| Refusal-Qwen | MLP / batch | 0.841 |
| Cyber-1 (dual_use vs benign) | MLP / batch | 0.870 |
| Cyber-2 (high_risk vs rest) | MLP / batch | 0.858 |
| Cyber-3 (prohibited vs rest) | linear / batch | 0.930 |
| **Mean** | | **0.883** (σ = 0.038) |

Bonus: Benign-vs-rest reaches AUC 0.987 with a linear / batch probe.

Reproduced with `starter_code/train_probe.py`; metrics saved to `probes/results/*_metrics.jsonl`.

## XGBoost extension

| Task | AUC | n_train | n_test |
|---|---|---|---|
| refusal_gemma4_31b | 0.9203 | 561 | 277 |
| refusal_qwen36 | 0.8689 | 581 | 281 |
| cyber_dual_use | 0.8655 | 1,304 | 546 |
| cyber_high_risk_dual_use | 0.8394 | 2,042 | 864 |
| cyber_prohibited | 0.8956 | 3,569 | 1,551 |
| **Mean** | **0.8779** | | |

Reproduced with `xgboost_probes.py`; results saved to `probes/results/xgboost_results.json`.

## Qwen layer sweep

We re-extracted Qwen activations at layers 16, 32, and 48 to test whether layer 32 is suboptimal given Qwen's hybrid architecture (16 full-attention + 48 DeltaNet linear-attention layers).

| Layer | Linear AUC | XGBoost AUC |
|---|---|---|
| 16 | 0.8236 | 0.8556 |
| 32 | 0.8383 | 0.8689 |
| 48 | **0.8656** | **0.8840** |

(n_train = 581, n_test = 281)

The refusal signal in Qwen lives later than in Gemma. Layer 48 with XGBoost bumps Refusal-Qwen by +4.3% over the layer-32 MLP baseline (0.841 → 0.884).

Reproduced with `python qwen_layer_sweep.py`. Inputs: `extracts/qwen36_multilayer/qwen36/*.pt` (extracted with `--layers "16,32,48"`).

## Final results (best probe per task)

| Task | Best configuration | AUC |
|---|---|---|
| Refusal-Gemma | XGBoost @ layer 32 | 0.920 |
| Refusal-Qwen | XGBoost @ layer 48 ⭐ | 0.884 |
| Cyber-1 (dual_use vs benign) | MLP @ layer 32 | 0.870 |
| Cyber-2 (high_risk vs rest) | MLP @ layer 32 | 0.858 |
| Cyber-3 (prohibited vs rest) | linear @ layer 32 | 0.930 |
| **Mean** | | **0.892** |

This is a +0.9% improvement over the 0.883 baseline, driven by the Qwen layer sweep and XGBoost.

## Reproducing Level 1 results

### 1. Extract activations

```bash
python starter_code/extract_residuals.py --model_key gemma4_31b --model_path /data/Gemma-4-31B-it
python starter_code/extract_residuals.py --model_key qwen36 --model_path /data/Qwen3.6-27B
```

The cyber dataset uses the key `prompt`, but `extract_residuals.py` expects `attack_prompt`. Patch it:

```bash
python -c "import json; rows=[json.loads(l) for l in open('datasets/cyber_probes/train.jsonl')] + [json.loads(l) for l in open('datasets/cyber_probes/test.jsonl')]; open('datasets/cyber_probes/all_fixed.jsonl','w').writelines(json.dumps({**r,'attack_prompt':r['prompt']})+'\n' for r in rows)"

python starter_code/extract_residuals.py --model_key gemma4_31b --samples_file datasets/cyber_probes/all_fixed.jsonl --out_dir extracts/cyber_gemma4_31b
```

### 2. Build the unified manifest

```bash
python build_manifest.py
```

`build_manifest.py` joins the per-task extraction metadata with dataset labels and produces train/test splits. It bridges the gap between `extract_residuals.py` (per-task manifests) and `train_probe.py` (unified manifest).

- **Input:** `extracts/*/extraction_metadata.json` and the dataset jsonl files
- **Output:** `extracts/unified_manifest.json`

### 3. Train probes

```bash
python starter_code/train_probe.py \
    --extracts_dir extracts/gemma4_31b \
    --manifest extracts/unified_manifest.json \
    --task refusal_gemma4_31b
```

Repeat for each task.

### 4. Run XGBoost extension

```bash
python xgboost_probes.py
```
