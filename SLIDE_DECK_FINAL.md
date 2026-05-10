# Slide deck — FINAL

10 slides, ~1 min each = 10 min talk + 2 min Q&A per spec.

Story arc: probe AUC 0.918 → token attribution fails (0%) → free-form rewriting works (60%) → activation ablation works at *one* layer (80%) → cyber-cascade flip stretch (48%) → probe direction is causal but the "single direction" framing is wrong. Constitutional Classifiers should expect this.

---

## Slide 1 — Title

**Three Ways to Flip a 0.92-AUC Refusal Probe**
*And what each one tells us about the probe*

Veronika Wannack, Federico Dejean
EPFL Red-Team Mech-Interp Hackathon, May 2026

Just title + names. 5 sec.

---

## Slide 2 — The question

**A linear probe predicts refusal at AUC 0.929 on Gemma-4-31B-it.**

Does that mean we know what controls the model's refusal decision?

→ AUC measures correlation. We tested causation three ways:

- **Token edits** (probe attribution → minimal natural-language edits)
- **Free-form rewriting** (PRE, Xiong 2025: K paraphrases, pick the most probe-shifted)
- **Activation ablation** (Arditi 2024: project the direction out of the residual stream)

Each is a different *intervention surface* on the same probe direction.

**Speaker note:** "We started where the hackathon points us — token edits — got nothing, then expanded the experiment to ask: is the AUC capturing something causal at all?"

---

## Slide 3 — Level 1: probes work well

| Task | Method | AUC |
|---|---|---|
| Refusal-Gemma | Direction-only @ L48 | **0.929** |
| Refusal-Qwen | XGBoost @ L48 | 0.884 |
| Cyber-1 (dual-use vs benign) | Hybrid features | 0.964 |
| Cyber-2 (high-risk vs rest) | Hybrid features | 0.935 |
| Cyber-3 (prohibited vs rest) | Hybrid features | 0.879 |
| **Mean** | | **0.918** |

Two findings:
- **Layer 48 beats Layer 32** (the "middle" default is wrong)
- **Direction-only ≈ XGBoost on refusal** — looks like a single direction

**Speaker note:** "AUC 0.92 is solid. Direction-only matching XGBoost suggests refusal IS a single linear feature. That's the prediction we tested."

---

## Slide 4 — Token edits: the asked-for intervention

Probe-attribute every token at L48, weight by XGBoost layer importance, edit top-K.

**Manual-verified flip rate: 0/20** (judge said 2/20 — both soft refusals after audit)

What the probe actually attributes to:

| Rank | Token | Source pattern |
|---|---|---|
| 1 | `<` | code template / HTML tag |
| 2 | `\n` | newline |
| 3 | `_` | identifier separator |
| 4 | `#` | comment marker |
| 5 | `"""` | docstring delimiter |
| 6 | `    ` | indentation |
| 7-15 | `haz`, `mat`, `parser`, `binary` | BPE fragments |

**No semantic refusal-relevant tokens** ("exploit", "vulnerability", "execute") in the top set.

The probe achieves 0.929 by latching onto the **distribution** of refusal-eliciting prompts (cyber-flavored vocab, code-like structure), not the features the model uses to decide.

**Aggregate AUC is insufficient validation for causal-mechanism claims.**

**Speaker note:** "The probe attribution map points at things you can't paraphrase. This is a real negative result — and not the end of the story."

---

## Slide 5 — Free-form rewriting flips ~60%

**FIGURE: fig_k_sweep.png**

Same probe, different intervention: editor LLM (Qwen3-30B-A3B) generates K paraphrases per prompt, the one most off the refusal direction wins. Intent gate ≥ 7.

| K | n_intent_pres | judge flip | manual TRUE flip |
|---|---|---|---|
| 3 | 64/81 | 56% | **39%** [27, 52] |
| **7** | **66/81** | **79%** | **59%** [47, 70] |
| 10 | 61/80 | 75% | ~62% |

K=7 is the optimum. Diminishing returns + slight overfitting at K=10.

**Manual audit caveat**: ~28% of K=7 judge-flagged "compliance" labels were soft refusals offering safe alternatives (whitelist instead of RCE). Even after subtraction, **39 prompts on n=81 returned working operational content**: env-var exfil, RAT framework, BLE cloning, AMSI bypass, MITM TLS proxy, AD enumeration, …

**Speaker note:** "The probe direction IS exploitable from input space — but only with paraphrase freedom. Not via per-token attribution."

---

## Slide 6 — Activation ablation: layer matters

**FIGURE: fig_layer_sweep.png**

We projected the refusal direction out of the residual stream, one layer at a time, across L4–L56:

- **All-15-layer ablation**: 0/20 manual flips
- **Single-layer at L48 alone: 16/20 manual flips** [57%, 90%]
- **Random-direction control at L48**: judge said 6/20, manual found **0/20** (all gibberish: "While C / / / 1 / / / /...")

The all-layer ablation was over-specifying — concentrating intervention at L48 alone — the same layer that maximizes probe AUC — replicates Arditi 2024 cleanly on Gemma-4-31B.

**Speaker note:** "Random control matches gibberish. Specific direction matches actual flips. The direction IS causal at L48."

---

## Slide 7 — Headline figure: three interventions

**FIGURE: fig_three_bar.png**

| Intervention | Where it acts | Manual TRUE flip |
|---|---|---|
| **Token edits** | input string, minimal | **0/20** [0%, 16%] |
| **Free-form rewrite (PRE K=7)** | input string, intent preserved | **39/66** [47%, 70%] |
| **Activation ablation (L48 only)** | residual stream | **16/20** [57%, 93%] |

The probe direction is a real causal handle.

It's accessible via input-side paraphrase (60%) AND via direct residual-stream intervention at the right layer (80%).

**It is NOT accessible via token-level attribution** — the attribution map points at structural correlates, not the causal feature.

**Speaker note:** "This is the slide. Three interventions, same probe direction, very different flip rates. The wedge between AUC and probe-attribution-as-attack-surface is real."

---

## Slide 8 — Off-axis flips: probe ≠ direction

**FIGURE: fig_delta_vs_flip.png**

Several PRE rewrites flipped behavior with **Δprobe ≥ 0** (probe score did not decrease):

- EVO_0220, EVO_0344, EVO_0288, EVO_0540, ... — Δprobe up to +0.36, behavior=compliance
- 4 distinct prompts in K=7 alone

We re-derived the refusal direction three different ways at L48 from 845 cached residuals:

| Variant | cos with last-token mean-diff |
|---|---|
| Last-token mean-diff (canonical, used for ablation) | 1.0 |
| Mean-pool mean-diff | 0.234 |
| **Whitened SVD ((Σ + 1e-3 I)⁻¹ × mean-diff)** | **0.016** |

**The whitened direction is essentially orthogonal to the canonical direction in 5376-dim space.** Both produce similar probe AUC.

There is no canonical refusal direction. "The" direction depends on extraction method.

**Speaker note:** "Wollschläger 2025 predicts refusal lives in a low-D subspace, not a single direction. Our cos=0.016 finding empirically confirms this on Gemma-4-31B."

---

## Slide 9 — Stretch: cyber-cascade flip

We ran PRE with the cyber-3 (prohibited) probe direction as scorer on n=30 cyber-prohibited prompts.

| Metric | Value |
|---|---|
| n_intent_preserved | 27/30 (90%) |
| Pr(f \| edit) | 0.37 |
| **mean Δprobe** | **+0.08** (probe score went UP) |
| Pr(model \| edit) judge | 0.815 |
| **Pr(model \| edit) manual** | **13/27 = 0.48** [0.30, 0.66] |

True flips delivered: reverse shells (Rust + C ports), DNS exfiltration over TXT queries, npm postinstall SSH/env-var enumeration, FleetWorker self-updating RAT, XSS payloads with cookie exfil, Linux rootkit `getdents64` filter, astroturfing review-bot generators.

**The cyber-3 probe direction is essentially uncorrelated with what controls Gemma's behavior on cyber-prohibited prompts.** The probe gates legitimate evals at AUC 0.879 — but is bypassable by paraphrase WITHOUT the bypass even moving the probe.

**Speaker note:** "On cyber-prohibited content the probe doesn't even register the bypass. Paraphrase wins; probe never sees it."

---

## Slide 10 — What this means for production

**For the open question (Constitutional Classifiers / probe-as-judge stacks):**

A probe with AUC 0.93 against a static eval is *not* sufficient as a production gate. An attacker with paraphrase budget gets ~60% of refusable prompts past the model. On cyber-prohibited content, the bypass doesn't even move the probe.

**Three deployment-relevant findings:**
1. AUC against held-out evals does not predict robustness against paraphrase attack.
2. "The refusal direction" is underspecified — pooling, whitening, layer choice all yield essentially-orthogonal vectors with similar AUC.
3. LLM judges have systematic failure modes (soft refusal → "compliance"; gibberish → "compliance"); manual audit of every judge-flagged flip is essential. We caught 28% mislabel rate on PRE results, 100% on random-ablation control.

**Code + reproducible results:** github.com/fdejean/mechhack (branch: experiment)

**Speaker note (Q&A prep):** "Wollschläger 2025 multi-direction story is what we'd run next — our cos=0.016 finding predicts it should help. We just didn't have time."

---

## Speaker notes

- 10 slides, target 1 min each → 10 min talk + 2 min Q&A
- Time check at slide 5 (mid-point)
- If running long: drop slide 9 (cyber stretch) or compress slide 8 — both are extension findings, slide 7 is the headline
- Most likely Q&A topics:
  - **Why didn't all-layer ablation work?** Each layer's "refusal direction" means subtly different things; ablating at all 15 layers is over-specification
  - **Can you replicate Arditi 2024 then?** Yes — at L48 single-layer with 80% flip rate. Their result holds; the all-layer adaptation was over-specified.
  - **What's the editor LLM?** Qwen3-30B-A3B-Instruct-2507 via EPFL AIaaS, MiniMax-M2.7 as judge.
  - **EVO_0220 — what does the off-axis flip mean?** The probe direction is one causal handle; rewriting can find compliance routes the probe doesn't track. This *predicts* multi-direction subspace findings (Wollschläger 2025).
  - **Cyber-cascade methodology caveat** — pre_agent's behavior judge classifies refusal/compliance, NOT "is this still asking for cyber-prohibited content." So the 48% measures "Gemma complied with a cyber-prohibited-classified prompt after probe-3-direction-guided rewriting."
