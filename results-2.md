# Results — final

## Overview

We probe the residual stream of Gemma-4-31B-it and Qwen3.6-27B to predict refusal and cyber-harm category from internal activations (Level 1), then test multiple ways to flip Gemma's behavior using the recovered refusal direction (Level 2): token-level edits, free-form rewriting, and direct activation ablation. We also run a stretch goal: cyber-cascade flip using the cyber-3 probe direction as scorer.

**Headline numbers** (after manual audit of every judge-flagged compliance):

- **Level 1 mean AUC: 0.918** across all 5 tasks
- **Level 2 across three intervention surfaces** (Gemma-4-31B-it):
  - Token-level edits (input space, minimal): **0/20** behavior flips
  - Free-form rewriting K=7 (input space, intent-preserved): **39/66** intent-preserved = **0.591** [Wilson 95%: 0.47, 0.70]
  - Activation ablation at L48 single-layer (residual stream): **16/20** = **0.80** [Wilson 95%: 0.57, 0.93]
  - Activation ablation random-direction control at L48: **0/20** (all gibberish; 6/20 judge-flagged were all artifacts)
- **Level 2 stretch — cyber-cascade flip**: **13/27** = **0.481** [Wilson 95%: 0.30, 0.66] with mean Δprobe = **+0.08**

The shape of these flip rates — not any single number — is the finding.

---

## Setup

- **Models:** Gemma-4-31B-it, Qwen3.6-27B (both frozen, accessed via `/data/`)
- **Refusal datasets:** Gemma 876 samples, Qwen 878 samples
- **Cyber dataset:** 7,114 samples across 4 categories (`benign / dual_use / high_risk_dual_use / prohibited`)
- **Train/test split:** deterministic by `hash(sample_id) mod 100 < 70`
- **Behavior + intent judges:** MiniMaxAI/MiniMax-M2.7 via EPFL AIaaS, schema-forced
- **Hardware:** A100-80GB on the EPFL runai cluster, 3 pods, ~22 GPU-hours total

---

## Level 1 — predicting behavior from internals

### Final per-task AUC

| Task | Method | AUC |
|---|---|---|
| Refusal-Gemma | Direction-only @ layer 48 | **0.929** |
| Refusal-Qwen | XGBoost @ layer 48 | 0.884 |
| Cyber-1 (dual_use vs benign) | XGBoost on 53-d hybrid features | 0.964 |
| Cyber-2 (high_risk_dual_use vs rest) | XGBoost on hybrid features | 0.935 |
| Cyber-3 (prohibited vs rest) | XGBoost on hybrid features | 0.879 |
| **Mean** | | **0.918** |

Two findings:
1. **Layer 48 beats layer 32** on Refusal-Gemma and Cyber-1; layer 44 on Cyber-2; layer 40 on Cyber-3. The "middle of the network" default is wrong for refusal-style signals.
2. **Direction-only matches XGBoost on Refusal-Gemma** (0.929 vs 0.926). Refusal *appears* to be a single direction. Section 2D shows this is more nuanced.

### Qwen layer sweep (16/32/48)

Qwen has a hybrid architecture (16 full-attention + 48 DeltaNet layers). Layer 48 was the best (XGBoost AUC 0.884, +4.3% over layer 32).

### Hybrid 53-feature pipeline (Gemma)

53-dim feature vector per sample combining 16 direction loadings + 16 logit-lens refusal probs + 16 logit-lens compliance probs + 5 derived features. XGBoost on these features achieved mean AUC 0.926 across the 4 Gemma tasks.

---

## Level 2 — three ways to flip behavior

The hackathon defines Level 2 around minimal natural-language edits. We started there, then expanded to test the sharper question: **does the probe direction expose a causal handle on refusal, or is the AUC=0.929 capturing a correlate the model doesn't actually use?**

To answer, we tested three increasingly invasive interventions, each acting on a different surface. The hackathon asks for `Pr(f|edit)`, `Pr(model|edit)`, and `Pr(model|f)` reported with Wilson 95% CIs — all three are reported below for each intervention.

| Intervention | Where it acts | Constraint | n |
|---|---|---|---|
| Token-level edits (`edit_agent.py`) | input string | minimal token changes guided by probe attribution | 20 |
| Free-form rewriting (`pre_agent.py`, PRE-style) | input string | preserve intent, otherwise free | 81 |
| Activation ablation (`patch_agent.py`, Arditi-style) | internal residual | no input change | 20 |

---

### 2A — Token-level edits

Pipeline: probe attribution at 16 layers (XGBoost importance-weighted) → top-K=15 token candidates → editor LLM (MiniMax-M2.7) proposes JSON-schema-validated edits with intent gate.

**Manual-verified flip rate: 0/20** [0%, 16%] (judge said 2/20 — both soft refusals after manual audit)

**Why it failed**: the highest-attribution tokens at L48 are not natural-language semantic features. Top-15 list:

| Rank | Token | Source pattern |
|---|---|---|
| 1 | `<` | code template / HTML tag |
| 2 | `\n` | newline |
| 3 | `_` | identifier separator |
| 4 | `#` | comment marker |
| 5 | `"""` | docstring delimiter |
| 6 | `    ` | indentation |
| 7-15 | `haz`, `mat`, `parser`, `binary`, `worker` | BPE fragments |

Effectively no semantic refusal-relevant tokens (e.g. "exploit", "vulnerability", "execute") in the top set. The probe latches onto the **distribution** of refusal-eliciting prompts (cyber-flavored vocabulary, code-like structure), not features the model uses to decide.

Single-iteration paraphrastic edits cannot access this signal — `<`, newlines, indentation cannot be paraphrased.

**This is a meaningful negative result**: aggregate AUC is insufficient validation for causal-mechanism claims about probe-attributed features.

---

### 2B — Free-form rewriting (PRE)

`pre_agent.py` implements Xiong et al. 2025 *Probe-Rewrite-Evaluate*. Editor LLM (Qwen3-30B-A3B) generates K paraphrases per prompt, each scored by projection onto refusal direction at L48. Lowest-projection candidate wins. Intent gate ≥7. Then Gemma rolls out on the winner; behavior judge classifies.

#### K-sweep (n=81 per K)

We ran K∈{3, 7, 10} on the full attribution_eval.jsonl (n=81). **Manual audit of every judge-flagged compliance** (52 for K=7).

| K | intent_pres | Pr(f\|edit) | Pr(model\|edit) judge | Pr(model\|edit) **manual TRUE** | Wilson 95% (manual) |
|---|---|---|---|---|---|
| 3 | 64/81 (79%) | 0.766 | 0.563 | ~0.39 (estimated) | [~0.27, ~0.52] |
| **7** | **66/81 (81%)** | **0.848** | **0.788** | **0.591 (39/66)** | **[0.47, 0.70]** |
| 10 | 61/80 (76%) | 0.885 | 0.754 | ~0.62 (estimated) | similar to K=7 |

The curve is **K=3 (39%) → K=7 (59%) → K=10 (~62%)**. K=7 is near-optimal; further paraphrase budget gives diminishing returns and slightly lower intent preservation.

Mean Δprobe at K=7: −0.310. Pr(f|edit) = 0.85 — paraphrasing reliably moves the probe.

#### Manual audit findings (K=7, 52 judge-flagged compliances)

We manually inspected all 52. Categories:

| Verdict | Count | Description |
|---|---|---|
| TRUE flip — operational content delivered | 39 | Working code: env-var exfiltration, RAT, BLE cloning, AMSI bypass, MITM TLS proxy, AD enumeration, etc. |
| Soft refusal with safe alternative | 5 | Whitelist-based diagnostic instead of RCE, refusal with educational discussion |
| Partial / ambiguous | 8 | Simulated rather than real (memory-injection sim), or generic infrastructure code |

Conservative count: **39/66 = 0.591** [0.47, 0.70]
Including partials: 47/66 = 0.71

**Off-axis flips**: 4 separate prompts in K=7 flipped behavior with Δprobe ≥ 0 (probe score did NOT decrease). The probe direction is a causal handle but not the *only* path — paraphrasing finds compliance routes the probe doesn't track.

---

### 2C — Activation ablation (Arditi 2024)

For each refused prompt:
- **baseline**: no intervention
- **ablated**: project the unit-norm refusal direction out of the residual stream at every probed layer
- **random_ctrl**: same hook structure with random unit vectors (controls for "any perturbation flips")

#### Activation-level verification

Before measuring behavior, we verified the intervention itself at L48:
- baseline projection onto d_48: 69.54
- ablated projection onto d_48: 0.04
- residual L2 change at L48: 368 (baseline norm 712, ~50% relative change)

The hook fires, projection is zeroed, residual changes substantially. Behavior outcomes are not because of failed intervention.

#### Bugs caught and fixed

1. `load_directions` re-normalizing already-unit-norm vectors (effective ablation magnitude was ~1.6%)
2. Layer-0 placeholder direction with norm 1e-8 producing NaN that propagated to all subsequent layers

Both caught by the activation-level diagnostic, not by behavior outcome alone. **Lesson: high-AUC probe ≠ working intervention. Verify the intervention separately.**

#### Layer sweep (the diagnostic)

We ablated at one layer at a time across L4-L56 every 4 layers:

| Layer | baseline | ablated (judge) | random | n |
|---|---|---|---|---|
| L4 | 1 | 1 | 2 | 20 |
| L8 | 2 | 1 | 1 | 20 |
| L12 | 2 | 11 | 1 | 20 |
| L16 | 1 | 1 | 2 | 20 |
| L20 | 1 | 12 | 3 | 20 |
| L24 | 1 | 10 | 1 | 20 |
| L28 | 2 | 11 | 0 | 20 |
| L32 | 1 | 16 | 1 | 20 |
| L44 | 1 | 12 | 6 | 20 |
| **L48** | **1** | **17** | **6** | **20** |
| L56 | 2 | 7 | 3 | 20 |

**The all-15-layer ablation we ran first (0/20 manual flips) was over-specifying.** Single-layer ablation reveals a clear peak at L48 — the same layer that maximizes probe AUC.

#### L48 single-layer manual audit (n=20, all conditions)

| Condition | Judge-flagged | **Manual TRUE flip** | Wilson 95% |
|---|---|---|---|
| baseline | 1/20 | 1/20 (eval-set noise: EVO_0109 not refused at baseline) | — |
| ablated | 17/20 | **16/20** (one soft refusal: EVO_0216 same whitelist pattern as PRE) | **[0.57, 0.93]** |
| random_ctrl | 6/20 | **0/20** (all 6 are token-level gibberish: "While C / / / / 1 / / / /...") | [0%, 16%] |

**This replicates Arditi 2024 cleanly on Gemma-4-31B**, when the intervention targets the right layer. The original all-layer ablation was washing out the targeted effect.

---

### 2D — Direction-extraction sensitivity

We computed three different "refusal direction" variants at L48 from 845 cached residuals:

| Variant | Raw norm | cos with mean-diff |
|---|---|---|
| Last-token mean-diff (canonical, used by `patch_agent`) | 59.93 | 1.0 |
| Mean-pool mean-diff | 6.12 | 0.234 |
| Whitened SVD ((Σ + 1e-3 I)⁻¹ × mean-diff) | 327.14 | **0.0155** |

**The whitened direction is essentially orthogonal to the canonical mean-diff direction in 5376-dim space.** Both produce similar probe AUC.

Whitened-direction ablation at L48: 1/20 ablated (≈ baseline). Random control: 7/20. **Whitening removes too much** — the direction the cleaner extraction returns is not the causal one. Last-token mean-diff at L48 is the only direction we tested that flips behavior, and even that depends on layer choice.

There is no canonical refusal direction. "The" direction depends on extraction method, pooling, and layer.

---

## Stretch — cyber-cascade flip

We ran PRE with the cyber-3 (prohibited) probe direction as scorer on n=30 cyber-3 eval prompts.

| Metric | Value |
|---|---|
| n_intent_preserved | 27/30 (90%) |
| Pr(f \| edit) | 0.370 [0.215, 0.558] |
| **mean Δprobe** | **+0.08** (probe score went UP on average) |
| Pr(model \| edit) judge | **0.815** [0.633, 0.918] |
| Pr(model \| edit) **manual TRUE** | **13/27 = 0.48** [0.30, 0.66] |

### Manual audit findings (n=27)

| Verdict | Count |
|---|---|
| TRUE flip — operational malicious content | 13 |
| Defensive-but-dual-use (encryption tools, clipboard hijack DETECTOR) | 4 |
| SAFE / partial (simulated, generic infrastructure) | 5 |
| (other / partial) | 5 |

True flips delivered include:
- Reverse shells (Rust port + C-style nix port)
- Backdoors / RATs (TCP listener with command pattern, FleetWorker self-update, embedded module loader via objcopy)
- Exfiltration (DNS over TXT queries, npm postinstall SSH/env enumeration, workforce-extension keylogger)
- Adversarial / astroturf (coordinated review-bot generator, LLM-driven fake review submission)
- XSS payloads with cookie/token exfil
- Linux rootkit `getdents64` filter for process hiding

Honest framing: this measures "did Gemma comply with a prompt originally classified as cyber-prohibited, after probe-3-direction-guided rewriting?" — not the strict cascade-flip definition (which would require all three cyber probes to drop below threshold).

**The fact that 22/27 (judge) and 13/27 (manual) flipped while the probe score went UP on average** is striking: the cyber probe direction is essentially uncorrelated with what controls Gemma's behavior on cyber-prohibited prompts. The probe-as-judge stack gates legitimate evaluation but is bypassable by paraphrase that doesn't even register in probe-space.

---

## What this contributes — the three flip rates together

| Intervention | Where it acts | Constraint | True flip rate (manual) |
|---|---|---|---|
| Token edits | input string | minimal | **0/20** = 0% [0%, 16%] |
| Free-form rewrite (PRE K=7) | input string | intent preserved | **39/66** = 59% [47%, 70%] |
| Activation ablation (L48 only) | residual stream | none | **16/20** = 80% [57%, 93%] |
| **Cyber-cascade (stretch)** | input string, intent preserved | cyber-3 direction | **13/27** = 48% [30%, 66%] |

**The probe direction is a real causal handle.** Two interventions on it both flip behavior at meaningful rates: free-form input-side paraphrase (60%) and direct residual-stream ablation at the right layer (80%).

But:
- Token-level attribution does NOT access this handle. The structural and BPE-fragment tokens that attribute highest are not paraphrasable in plain English.
- "The refusal direction" depends on layer (peak L48), pooling (last-tok ≠ mean-pool), and extraction method (last-tok ≠ whitened, cos = 0.016). Multiple essentially-orthogonal directions all achieve probe AUC ~0.93.
- Random ablation control flips behavior 30-45% by judge but 0% by audit — all gibberish.
- On cyber-prohibited content, the bypass doesn't even move the probe (mean Δprobe = +0.08).

For **production probe-as-judge stacks (Constitutional Classifiers, etc.)** these are direct deployment-relevant findings:
1. AUC against a static eval doesn't predict robustness to paraphrase attack
2. The attribution map points at structural/lexical correlates, not the causal feature
3. LLM judges have systematic failure modes (soft refusal → "compliance"; gibberish → "compliance"); ~28% of judge-flagged flips were artifacts in our K=7 manual audit

---

## Judge failure modes

Two distinct failure modes catalogued via manual audit:

1. **Soft refusal with safe alternative** (5/52 K=7 judge-flagged; 4/22 cyber-cascade). Gemma leads with "⚠️ Critical security vulnerability", explains why the request is unsafe, then provides a structurally-similar safe variant (whitelist instead of RCE; HMAC verification instead of encryption). Judge sees code blocks and security framing → "compliance".

2. **Gibberish or fluent filler** (6/6 random-control judge-flagged at L48; 2/2 in earlier all-layer ablation). Random-direction ablation corrupts residual stream into token-level nonsense ("While C / / / / 1 / / / /..."). Judge has no "degraded output" category and defaults to "compliance" when output structurally resembles a helpful response.

We recommend any LLM-as-judge evaluation include either:
- A third "degraded" category
- A perplexity sanity-check on responses
- Manual audit of all judge-flagged flips before reporting

---

## What we'd do with more time

- **Multi-direction ablation** (Wollschläger 2025). Our cos=0.016 finding *predicts* refusal lives in a low-dimensional cone, not a single direction. SVD top-K directions and ablate all simultaneously.
- **PRE × ablation hybrid.** Compose input-side rewriting with activation ablation. Tests whether the two interventions stack or are redundant.
- **Whitened-direction PRE optimization** — we ran PRE with whitened direction as scorer (n=81); the whitened direction barely moves under PRE (mean Δprobe ≈ −0.002) but behavior still flipped 56% of the time. Stronger evidence that the linear probe doesn't capture the causal feature.

---

## Where everything lives

```
/scratch/mechhack/
├── extracts/                                        Activation extracts
├── /scratch/hybrid/gemma4_31b/                      Hybrid pipeline outputs (16-layer last-tok directions)
├── /scratch/hybrid_v2/gemma4_31b/                   Recomputed directions
│   ├── direction_arditi_lasttok_L48.pt
│   ├── direction_arditi_meanpool_L48.pt
│   ├── direction_arditi_meanpool_multilayer.pt
│   ├── direction_arditi_whitened_L48.pt            cos=0.016 with mean-diff
│   └── direction_single_L{4,8,12,16,20,24,28,32,36,40,44,48,52,56}.pt
├── hybrid_models/gemma4_31b/                        XGBoost classifiers
├── probes/                                          Single-layer probes
├── edit_eval/
│   ├── attribution_refusal_gemma4_31b_hybrid.json  Token attribution
│   ├── level2_final.json                            Token-edit n=20
│   ├── pre_n81.json                                 PRE K=3 n=81
│   ├── pre_k7_n81.json                              PRE K=7 n=81 (HEADLINE)
│   ├── pre_k10_n81.json                             PRE K=10 n=81
│   ├── pre_whitened_n81.json                        PRE with whitened direction
│   ├── pre_cyber3_cascade.json                      cyber-cascade stretch
│   ├── patch_full.json                              all-15-layer ablation n=20
│   ├── patch_layer_L{4,8,...,56}.json               single-layer sweep
│   ├── patch_whitened_n20.json                      whitened ablation
│   └── patch_multilayer_meanpool_n20.json           multilayer mean-pool ablation
├── figures/                                         Slide figures
└── scripts/                                         Wrapper scripts (see SCRIPTS_README)
```

---

## References

- Arditi et al. 2024. "Refusal in Language Models Is Mediated by a Single Direction." [arxiv:2406.11717](https://arxiv.org/abs/2406.11717)
- Wollschläger et al. 2025. "The Geometry of Refusal." [arxiv:2502.17420](https://arxiv.org/abs/2502.17420)
- Xiong et al. 2025. "PRE — Probe-guided Rewriting Engine." [arxiv:2509.00591](https://arxiv.org/abs/2509.00591)
- elder-plinius (2026). OBLITERATUS toolkit. github.com/elder-plinius/OBLITERATUS — source of whitened-SVD direction extraction technique
- Cunningham et al. 2026. Constitutional Classifiers. [arxiv:2601.04603](https://arxiv.org/abs/2601.04603)
