Scaffolding code: 

build_manifest.py: joined the activation files with their labels.
Input: extracts/*/extraction_metadata.json + dataset jsonl files
Output: extracts/unified_manifest.json (labels + train/test splits)
Why needed: train_probe.py expects a unified manifest format, but extract_residuals.py produces simpler per-task manifests. You bridged the gap.


++ The all_fixed.jsonl data fix: the cyber dataset uses prompt, but the script wants attack_prompt


=====

## How to run

## Reproducing our Level 1 results

# 1. Extract activations (provided script):
python starter_code/extract_residuals.py --model_key gemma4_31b --model_path /data/Gemma-4-31B-it
python starter_code/extract_residuals.py --model_key qwen36 --model_path /data/Qwen3.6-27B
# Cyber needs attack_prompt key added (cyber jsonl uses 'prompt'):
python -c "import json; rows=[json.loads(l) for l in open('datasets/cyber_probes/train.jsonl')] + [json.loads(l) for l in open('datasets/cyber_probes/test.jsonl')]; open('datasets/cyber_probes/all_fixed.jsonl','w').writelines(json.dumps({**r,'attack_prompt':r['prompt']})+'\n' for r in rows)"
python starter_code/extract_residuals.py --model_key gemma4_31b --samples_file datasets/cyber_probes/all_fixed.jsonl --out_dir extracts/cyber_gemma4_31b

# 2. Build manifest:
python build_manifest.py

# 3. Train probes (provided script):
python starter_code/train_probe.py --extracts_dir extracts/gemma4_31b --manifest extracts/unified_manifest.json --task refusal_gemma4_31b
# (repeat for each task)

# 4. XGBoost extension:
python xgboost_probes.py

=====

## Results 

Level 1 — Probe AUC at layer 32, Gemma-4-31B
============================================================
  Refusal-Gemma                  attention/batch  AUC 0.914
  Refusal-Qwen                   mlp/batch        AUC 0.841
  Cyber-1 (dual_use vs benign)   mlp/batch        AUC 0.870
  Cyber-2 (high_risk vs rest)    mlp/batch        AUC 0.858
  Cyber-3 (prohibited vs rest)   linear/batch     AUC 0.930
============================================================
  MEAN AUC across 5 tasks: 0.883
  STDEV: 0.038

  How to get this: starter_code/train_probe.py (starter code, provided) 
  results: probes/results/*_metrics.jsonl

  Bonus — Benign-vs-rest:        linear/batch     AUC 0.987

Setup:
  - Activations from Gemma-4-31B layer 32 (middle of 64)
  - Refusal: Gemma 876 samples, Qwen 878 samples
  - Cyber:  7114 samples, 4 categories
  - Probe types: linear / MLP / attention / 4-head attention × 5 seeds × 2 regimes


XGBoost 

root@dev-pod-0-1:/scratch# python /scratch/mechhack/xgboost_probes.py
Task                           XGBoost AUC 
==================================================
refusal_gemma4_31b             0.9203     (n_train=561, n_test=277)
refusal_qwen36                 0.8689     (n_train=581, n_test=281)
cyber_dual_use                 0.8655     (n_train=1304, n_test=546)
cyber_high_risk_dual_use       0.8394     (n_train=2042, n_test=864)
cyber_prohibited               0.8956     (n_train=3569, n_test=1551)

Mean AUC: 
  0.8779


how to get it: xgboost_probes.py; output: probes/results/xgboost_results.json
