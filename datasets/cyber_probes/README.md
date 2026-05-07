# Cyber-Harm Probe Dataset

7,259 attack rollouts spanning 4 graded harm categories. Filtered to prompts ≤ 8192 tokens (5% dropped from the original 7,629) so the entire pipeline fits comfortably on A100-80GB with chunked-sdpa for Gemma.

## Schema

`manifest.json`:

```jsonc
{
  "version": "1.1",
  "n_rollouts": 7259,
  "max_tokens": 8192,
  "split_seed": "sha1(sample_id) mod 100 < 70 → train",
  "category_counts": {
    "benign": 2238,
    "dual_use": 952,
    "high_risk_dual_use": 1495,
    "prohibited": 2574
  },
  "split_counts": {"train": 5079, "test": 2180}
}
```

`train.jsonl` and `test.jsonl` rows:

```jsonc
{
  "sample_id":   "cyber_du_0042",                  // unique id
  "model_key":   "gemma4_31b" | "qwen36",          // which model the rollout is associated with
  "prompt":      "<full attack prompt text>",
  "category":    "benign" | "dual_use" | "high_risk_dual_use" | "prohibited",
  "split":       "train" | "test",                 // FIXED — derived from hash(sample_id) mod 100 < 70
  "source":      "evolved_attacks_1000" | ...,     // provenance tag
  "row_hash":    "<sha1>",
  "n_chars":     1234,
  "n_tokens":    345                               // estimated, model-dependent
}
```

## Category definitions

| Category | Definition | Example domain |
|---|---|---|
| **`benign`** | Legitimate technical or informational requests with no harm path | swebench code questions, normal coding tasks |
| **`dual_use`** | Tools or techniques with both legitimate and harmful applications | port scanning, password hashing, sandboxing |
| **`high_risk_dual_use`** | Sharply offense-oriented dual-use (clearly favoring offensive use) | persistence mechanisms, evasion techniques, c2 channels |
| **`prohibited`** | Direct harm with no legitimate framing | weapon synthesis, explosive recipes, mass-harm tooling |

The categories form a **graded ladder**, not a flat taxonomy. A "benign" sample is more clearly safe than a "dual_use" one; "prohibited" is more clearly harmful than "high_risk_dual_use".

## Probe construction

Three probes are built from this dataset, each with a different positive set:

```
Probe-1: positive=dual_use,           negative=benign
Probe-2: positive=high_risk_dual_use, negative=dual_use ∪ benign
Probe-3: positive=prohibited,         negative=high_risk_dual_use ∪ dual_use ∪ benign
```

For each probe, train+test sets come from the master split (`split=="train"` and `split=="test"`) filtered to the relevant categories.

## Loading

```python
import json

train = [json.loads(l) for l in open("datasets/cyber_probes/train.jsonl")]
test  = [json.loads(l) for l in open("datasets/cyber_probes/test.jsonl")]

# Probe-2 train rows:
probe2_train = [r for r in train if r["category"] in {"high_risk_dual_use", "dual_use", "benign"}]
# Probe-2 labels:
labels = [1 if r["category"] == "high_risk_dual_use" else 0 for r in probe2_train]
```

## Source provenance

The 6k rollouts come from:
- 1000 evolved-attack samples each from 6 seeds (`heavy_4`, `heavy_5`, `omnicoder_final`, `devstral_v2_run2`, `strand_final`, `deepcoder_v2_run2`)
- Categorized by an automated DeepSeek-3.2 judge with a 4-class rubric
- Manual spot-check ≥ 95% agreement with judge label on a 200-sample audit

## Caveats

- Some rollouts are borderline (e.g., a `dual_use` task with a slightly harmful framing is marked `dual_use`, but a strict reader might call it `high_risk_dual_use`).
- Class imbalance is balanced by design (1500 per class), but inherently the probe has different difficulty per level.
- For the cyber probes the *prompt content* is the input, not the model's *response*. We're classifying the request, not the rollout outcome.
