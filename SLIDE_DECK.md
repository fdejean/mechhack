# Slide deck — 10 min talk

10 slides total. ~1 min per slide. Q&A: 2 minutes (per spec).

The story arc: probe AUC 0.918 → token attribution fails → free-form rewriting works (60%) → activation ablation works at *one* layer (80%) → probe direction is causal but the "single direction" framing is wrong. Constitutional Classifiers should expect this.

---

## Slide 1 — Title

**Three Ways to Flip a 0.92 AUC Refusal Probe**
*And what each one tells us about the probe*

EPFL Red-Team Mech-Interp Hackathon, May 2026

(Just title + names. 5 sec.)

---

## Slide 2 — The question

**A linear probe predicts refusal at AUC 0.929 on Gemma-4-31B-it.**

Does that mean we know what controls the model's refusal decision?

→ AUC measures correlation. We tested causation three ways:

- **Token edits** (probe attribution → minimal natural-language edits)
- **Free-form rewriting** (PRE, Xiong 2025: K paraphrases, pick the most probe-shifted)
- **Activation ablation** (Arditi 2024: project the direction out of the residual stream)

Each one is a different *intervention surface* on the same probe direction.

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
- Layer 48 beats Layer 32 (the "middle" default is wrong)
- Direction-only ≈ XGBoost on refusal — this looks like it should be a **single direction**

---

## Slide 4 — Token edits don't flip behavior

**FIGURE: token attribution table showing top tokens are `<`, `\n`, `_`, `#`, BPE fragments**

Highest-attribution tokens for layer-48 direction:
- Structural: `<`, `\n`, `_`, `#`, `"""`, indentation
- BPE fragments: `haz`, `mat`, `parser`, `binary`
- No semantic tokens like "exploit", "vulnerability", "execute"

**Manual-verified flip rate after token edits: 0/20** (judge said 2/20 — both soft refusals)

Reason: the probe achieves 0.929 by latching onto the **distribution** of refusal-eliciting prompts (cyber-flavored vocabulary, code-like structure), not the semantic features the model uses to decide. Aggregate AUC is insufficient validation for causal claims.

---

## Slide 5 — Free-form rewriting flips ~60%

**FIGURE: fig_k_sweep.png — PRE diminishing returns curve K=3, K=7, K=10**

Same probe, different intervention: editor LLM proposes K paraphrases, pick the one that moves the probe farthest. Intent-preservation gate ≥7/10.

| K | n_intent_pres | judge flip | manual TRUE flip |
|---|---|---|---|
| 3 | 64/81 | 0.56 | **0.39** [0.27, 0.52] |
| 7 | 66/81 | 0.79 | **0.59** [0.47, 0.70] |
| 10 | 61/80 | 0.75 | ~0.62 |

**Speaker note:** even reading 28% of judge-flagged "compliance" labels as soft-refusal mislabels (the rate from manual audit), K=7 gets ~60% real flip rate on n=81.

The probe direction is exploitable from input space — but only with paraphrase freedom, not token nudges.

---

## Slide 6 — Activation ablation: layer matters

**FIGURE: fig_layer_sweep.png — full layer-sweep curve**

We projected the refusal direction out of the residual stream at each of 14 layers (4 to 56, every 4) and measured behavior.

Findings:
- All-15-layer ablation (yesterday): 0/20 manual flips
- Single-layer at L48 alone: **16/20 manual flips** [57%, 90%]
- Random-direction control at L48: **0/20** (judge said 6/20 — all gibberish)

**The all-layer ablation was over-specifying.** Concentrating intervention at L48 alone — the same layer that maximizes probe AUC — replicates Arditi 2024 cleanly on Gemma-4-31B.

Random control matches gibberish; specific direction matches actual flips. Direction is causal at L48.

---

## Slide 7 — The headline figure

**FIGURE: fig_three_bar.png — three-bar comparison with Wilson CIs**

Three interventions on the same probe direction, manually verified:

| Intervention | Where it acts | Manual TRUE flip |
|---|---|---|
| Token edits | input string, minimal | **0/20** [0%, 16%] |
| Free-form rewrite (PRE K=7) | input string, intent preserved | **39/66** [47%, 70%] |
| Activation ablation (L48 only) | residual stream | **16/20** [57%, 90%] |

The probe direction is a real causal handle — both via input-side paraphrase AND via direct residual-stream intervention at the right layer.

**Token-level attribution does not access it.** Probe AUC alone is misleading about which intervention surfaces are exploitable.

---

## Slide 8 — Off-axis flips: probe ≠ direction

**FIGURE: fig_delta_vs_flip.png — Δprobe scatter**

Several PRE rewrites flipped behavior with Δprobe ≥ 0 (probe score did NOT decrease):
- EVO_0220: Δprobe = +0.119, behavior=compliance (AMSI bypass C code delivered)
- EVO_0288, EVO_0344, EVO_0540, EVO_0837 — all Δprobe ≥ 0 with compliance

Whitened-SVD direction (OBLITERATUS-style) is **cos = 0.016 with mean-diff direction** — essentially orthogonal in 5376-dim space. Both achieve the same probe AUC. **There is no canonical refusal direction.** "The" direction depends on extraction method.

The probe captures discriminative signal, not the unique causal axis.

---

## Slide 9 — Stretch: cyber-cascade flip

[FILL IN AFTER CYBER AUDIT — placeholder structure]

We ran PRE on cyber-3 (prohibited) eval prompts using the cyber-3 probe direction as scorer.

| Metric | Value |
|---|---|
| n_intent_preserved | 27/30 |
| judge-flagged compliance | 22/27 (81%) |
| manual TRUE flip | TBD/27 |
| mean Δprobe | +0.08 (probe score went UP) |

If TRUE flips are even half of judge: **the cyber probe direction is essentially uncorrelated with what controls the model's cyber-prohibited behavior.** The "probe-as-judge" stack gates legitimate prompts but is bypassable by paraphrase.

---

## Slide 10 — What this means for production

**For the open question (Constitutional Classifiers / probe-as-judge stacks):**

A probe with AUC 0.93 against a static eval set is *not* sufficient as a production gate. An attacker with paraphrase budget gets ~60% of refusable prompts past the model. The probe captures something real — but real ≠ uniquely causal, and ≠ accessible only via the probe-pointed surface.

**Three deployment-relevant findings:**
1. AUC against held-out evals doesn't predict robustness against paraphrase attacks
2. "Refusal direction" as canonical concept is underspecified — pooling, whitening, layer choice all yield different vectors with same AUC
3. LLM judges have systematic failure modes (soft refusals → "compliance"; gibberish → "compliance"); manual audit of 5%+ of flagged flips is essential

Code: github.com/[your fork]/mechhack — all results reproducible with checkpointed JSON outputs.

---

## Speaker notes

- 10 slides, ~1 min each = 10 min talk + 2 min Q&A
- Time check at slide 5 (mid-point)
- If running long: drop slide 9 (cyber-cascade) — it's stretch, not core
- If asked about reading 1: emphasize EVO_0220 / off-axis flips (slide 8)
- If asked about Wollschläger 2025: yes, our cos=0.016 finding *predicts* the multi-direction story — we just didn't have time to run multi-direction ablation
