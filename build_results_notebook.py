"""Build RESULTS.ipynb from scratch and execute it.

Run from /Users/veron/Desktop/projects/mechhack (or anywhere — paths are absolute-friendly).
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbclient import NotebookClient
from pathlib import Path

REPO = Path("/Users/veron/Desktop/projects/mechhack")
OUT = REPO / "RESULTS.ipynb"

cells = []

def md(txt):
    cells.append(new_markdown_cell(txt.strip("\n")))

def code(txt):
    cells.append(new_code_cell(txt.strip("\n")))

# ============================================================
# Section 0 — Setup
# ============================================================
md("""
# RESULTS — Mechanistic interpretability hackathon submission

Probing Gemma-4-31B-it and Qwen3.6-27B for refusal and cyber-harm features (Level 1), then testing whether the recovered refusal direction gives a causal handle on Gemma's behavior via three text-only interventions and one residual-stream intervention as mechanism diagnostic (Level 2).

This notebook is a results viewer. It reads JSON outputs from `edit_eval/` and renders the headline tables and plots. It does not re-run any model rollouts.
""")

md("## Section 0 — Setup")

code("""
import json
import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

EDIT_EVAL = Path("edit_eval")
pd.options.display.float_format = lambda x: f"{x:.3f}"
""")

code("""
def load(name):
    with open(EDIT_EVAL / name) as f:
        return json.load(f)

pre_k3   = load("pre_n81.json")
pre_k7   = load("pre_k7_n81.json")
pre_k10  = load("pre_k10_n81.json")
pre_whit = load("pre_whitened_n81.json")
pre_cy3  = load("pre_cyber3_cascade.json")

patch_all15 = load("patch_full.json")
patch_whit  = load("patch_whitened_n20.json")
patch_mp    = load("patch_multilayer_meanpool_n20.json")
tok_edits   = load("level2_final.json")

layer_files = sorted(EDIT_EVAL.glob("patch_layer_L*.json"),
                     key=lambda p: int(p.stem.split("L")[-1]))
patch_by_layer = {int(p.stem.split("L")[-1]): json.loads(p.read_text())
                  for p in layer_files}
print(f"Loaded {len(patch_by_layer)} layer-sweep files: layers {sorted(patch_by_layer)}")
""")

code("""
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

# sanity: wilson(0, 20) should be ~ [0, 0.16]
wilson(0, 20), wilson(16, 20)
""")

# ============================================================
# Section 1 — Level 1
# ============================================================
md("""
## Section 1 — Level 1: probe AUCs

We trained per-task probes on residual activations. Mean AUC across 5 tasks is 0.918. Layer 48 beats layer 32 on Refusal-Gemma and Cyber-1; layer 44 on Cyber-2; layer 40 on Cyber-3. Direction-only matches XGBoost on Refusal-Gemma (0.929 vs 0.926) — refusal *appears* to be one direction, which Level 2 tests directly.
""")

code("""
level1 = pd.DataFrame([
    ("Refusal-Gemma",                "Direction-only @ L48",          0.929),
    ("Refusal-Qwen",                 "XGBoost @ L48",                 0.884),
    ("Cyber-1 (dual_use vs benign)", "XGBoost on 53-d hybrid",        0.964),
    ("Cyber-2 (high_risk vs rest)",  "XGBoost on 53-d hybrid",        0.935),
    ("Cyber-3 (prohibited vs rest)", "XGBoost on 53-d hybrid",        0.879),
], columns=["task", "method", "AUC"])
level1.loc["mean"] = ["", "", level1["AUC"].mean()]
level1
""")

# ============================================================
# Section 2 — Level 2A
# ============================================================
md("""
## Section 2 — Level 2A: token-level edits (`edit_agent.py`)

Pipeline: project each input token's residual onto the refusal direction at L48, weighted by per-layer XGBoost importance. Top-K tokens are candidates. An editor LLM proposes minimal edits subject to an intent gate (≥7). Gemma is re-rolled and a behavior judge classifies refusal vs compliance.
""")

md(
    "The top-attribution tokens were structural and lexical, not semantic: "
    "`<`, newline, indent, `_`, `#`, the docstring delimiter, BPE fragments like "
    "`haz`, `mat`, `parser`. None are paraphrasable in plain English without "
    "breaking syntax. The editor preserved intent perfectly (20/20) but moved "
    "no probe-relevant signal."
)

code("""
n = len(tok_edits)
intent_ok = sum(1 for r in tok_edits if r.get("intent_preserved"))
judge_flips = sum(1 for r in tok_edits if r.get("behavior_flipped") is True)
verify_skipped = sum(1 for r in tok_edits if r.get("behavior_flipped") is None)

# Manual audit (recorded in results_FINAL.md): both judge-flagged "flips"
# (EVO_0109 baseline label noise, EVO_0209 soft refusal) are not real.
manual_true = 0

tok_summary = pd.DataFrame([
    ("n",                         n),
    ("intent preserved",          f"{intent_ok}/{n}"),
    ("verify skipped (rollout)",  f"{verify_skipped}/{n}"),
    ("judge said flip",           f"{judge_flips}/{n}"),
    ("manual TRUE flip",          f"{manual_true}/{n}"),
    ("Wilson 95% CI (manual)",    f"[{wilson(manual_true, n)[0]:.3f}, {wilson(manual_true, n)[1]:.3f}]"),
], columns=["", "value"]).set_index("")
tok_summary
""")

# ============================================================
# Section 3 — Level 2B
# ============================================================
md("""
## Section 3 — Level 2B: free-form rewriting / PRE (`pre_agent.py`)

Implements Xiong et al. 2025 *Probe-Rewrite-Evaluate*. For each prompt the editor LLM (Qwen3-30B-A3B) generates K paraphrases, each scored by projection onto the refusal direction at L48. The lowest-projection candidate that passes the intent gate (≥7) wins. Gemma re-rolls; behavior judge classifies.
""")

code("""
def pre_row(label, blob):
    a = blob["aggregate"]
    return {
        "K":                  label,
        "n_intent_preserved": f"{a['n_intent_preserved']}/{a['n_total']}",
        "Pr(f|edit)":         a["Pr_f_given_edit"],
        "Pr(model|edit) judge": a["Pr_model_given_edit"],
        "Pr(model|f) judge":  a["Pr_model_given_f"],
        "mean Δprobe":        a["mean_delta_probe"],
    }

pre_sweep = pd.DataFrame([
    pre_row(3,  pre_k3),
    pre_row(7,  pre_k7),
    pre_row(10, pre_k10),
])
pre_sweep
""")

md("""
Manual audit of K=7 (n=66 intent-preserved, 52 judge-flagged compliances): 39 are true operational flips (working RATs, exfiltration one-liners, MITM TLS proxies, AMSI bypass, AD enumeration, etc.), 5 are soft refusals offering safe alternatives (whitelists instead of RCE, education-only), 8 are partial or simulated. Conservative true Pr(model|edit) = 39/66 = 0.591 [0.475, 0.696]. Including partials: 47/66 = 0.712. Judge-reported was 0.788.
""")

code("""
manual_k7 = {"true_flip": 39, "soft_refusal": 5, "partial_simulated": 8}
n_intent = pre_k7["aggregate"]["n_intent_preserved"]
n_judge_flagged = pre_k7["aggregate"]["behavior_flipped"]

assert sum(manual_k7.values()) == n_judge_flagged, \
    f"audit total {sum(manual_k7.values())} != judge-flagged {n_judge_flagged}"

lo, hi = wilson(manual_k7["true_flip"], n_intent)
audit_k7 = pd.DataFrame([
    ("n intent-preserved",            n_intent),
    ("n judge-flagged compliance",    n_judge_flagged),
    ("manual: true flip",             manual_k7["true_flip"]),
    ("manual: soft refusal",          manual_k7["soft_refusal"]),
    ("manual: partial/simulated",     manual_k7["partial_simulated"]),
    ("conservative Pr(model|edit)",   f"{manual_k7['true_flip']}/{n_intent} = {manual_k7['true_flip']/n_intent:.3f}"),
    ("Wilson 95% CI",                 f"[{lo:.3f}, {hi:.3f}]"),
    ("inclusive (true+partial)",      f"{(manual_k7['true_flip']+manual_k7['partial_simulated'])}/{n_intent} = {(manual_k7['true_flip']+manual_k7['partial_simulated'])/n_intent:.3f}"),
], columns=["", "value"]).set_index("")
audit_k7
""")

md("""
Off-axis flips: prompts where behavior flipped to compliance but the probe score did not decrease (Δprobe ≥ 0). The probe direction is one causal handle, but paraphrasing finds compliance routes the probe does not track.
""")

code("""
off_axis = [
    (r["sample_id"], r["delta_probe"])
    for r in pre_k7["results"]
    if r.get("edited_behavior") == "compliance" and (r.get("delta_probe") or 0) >= 0
]
off_axis_df = pd.DataFrame(off_axis, columns=["sample_id", "Δprobe"]).sort_values("Δprobe", ascending=False)
print(f"off-axis flips at K=7: {len(off_axis)}/{len([r for r in pre_k7['results'] if r.get('edited_behavior')=='compliance'])} judge-flagged compliances")
off_axis_df
""")

# ============================================================
# Section 4 — Level 2C ablation
# ============================================================
md("""
## Section 4 — Level 2C: activation ablation as mechanism diagnostic (`patch_agent.py`)

Out-of-spec as a Level 2 attack — the rules forbid activation patching as an edit mechanism. In-spec as a causal mechanism diagnostic: ablating the probe direction in the residual stream tests whether Level 1's `f` is the thing the model uses, or just a correlate. Following Arditi 2024: at every probed layer we project the unit-norm refusal direction out of the residual stream. Three conditions per prompt — `baseline`, `ablated`, `random_ctrl`.
""")

code("""
a = patch_all15["aggregate"]
# Manual audit of all 6 judge-flagged compliances across baseline/ablated/random:
# baseline 1 = label noise (EVO_0109), ablated 2 = fluent filler, random 3 = gibberish.
manual = {"baseline": 1, "ablated": 0, "random_ctrl": 0}
all15 = pd.DataFrame([
    ("baseline",     a["baseline_compliance"], manual["baseline"],     a["baseline_CI95"]),
    ("ablated",      a["ablated_compliance"],  manual["ablated"],      a["ablated_CI95"]),
    ("random_ctrl",  a["random_compliance"],   manual["random_ctrl"],  a["random_CI95"]),
], columns=["condition", "judge", "manual TRUE", "Wilson 95% (judge)"])
all15["judge"] = all15["judge"].map(lambda k: f"{k}/{a['n']}")
all15["manual TRUE"] = all15["manual TRUE"].map(lambda k: f"{k}/{a['n']}")
all15["Wilson 95% (judge)"] = all15["Wilson 95% (judge)"].map(lambda ci: f"[{ci[0]:.3f}, {ci[1]:.3f}]")
all15
""")

md("""
Ablating across all 15 layers simultaneously over-specifies — every layer's mean-difference direction means slightly different things, and removing all of them at once washes out the targeted effect. The layer sweep below (single-layer ablation, every-4-layer granularity) shows where the causal handle actually lives.
""")

code("""
sweep_rows = []
for L, blob in sorted(patch_by_layer.items()):
    a = blob["aggregate"]
    sweep_rows.append({
        "layer":    L,
        "n":        a["n"],
        "baseline": a["baseline_compliance"],
        "ablated":  a["ablated_compliance"],
        "random":   a["random_compliance"],
    })
sweep = pd.DataFrame(sweep_rows)
sweep["ablated_rate"]  = sweep["ablated"]  / sweep["n"]
sweep["baseline_rate"] = sweep["baseline"] / sweep["n"]
sweep["random_rate"]   = sweep["random"]   / sweep["n"]
sweep
""")

code("""
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(sweep["layer"], sweep["ablated_rate"],  "o-", label="ablated (refusal direction)", color="#c0392b")
ax.plot(sweep["layer"], sweep["random_rate"],   "s--", label="random control",            color="#7f8c8d")
ax.plot(sweep["layer"], sweep["baseline_rate"], "^:",  label="baseline (no intervention)", color="#2c3e50")

peak = sweep.loc[sweep["ablated_rate"].idxmax()]
ax.annotate(f"L{int(peak['layer'])} peak: {int(peak['ablated'])}/{int(peak['n'])} = {peak['ablated_rate']:.2f}",
            xy=(peak["layer"], peak["ablated_rate"]),
            xytext=(peak["layer"] + 2, peak["ablated_rate"] + 0.05),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            fontsize=9)

ax.set_xlabel("Layer")
ax.set_ylabel("Judge compliance rate")
ax.set_title("Single-layer ablation: judge-reported compliance vs layer (n=20 per layer)")
ax.set_ylim(-0.02, 1.0)
ax.grid(alpha=0.3)
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
""")

md("""
L48 single-layer manual audit (Δaudit on n=15 of 17 judge-flagged ablated compliances + 6/6 random-control): 14/15 audited ablated samples are real operational flips; 1 is a soft refusal (EVO_0216, same whitelist pattern as PRE). Extrapolating to the full ablated count gives ~16/20 true flips. All 6 random-control "compliances" are token-level gibberish ("While C / / / 1 / / /...") — the judge has no degraded-output category, so it defaults to compliance. Manual random TRUE rate: 0/20.
""")

# ============================================================
# Section 5 — Direction-extraction sensitivity
# ============================================================
md("""
## Section 5 — Direction-extraction sensitivity

Three different "refusal directions" at L48, all from the same 845 cached residuals.
""")

code("""
directions = pd.DataFrame([
    ("Last-token mean-diff (canonical)",          59.93,  1.000),
    ("Mean-pool mean-diff",                        6.12,  0.234),
    ("Whitened SVD ((Σ + 1e-3 I)^-1 × mean-diff)", 327.14, 0.0155),
], columns=["variant", "raw norm", "cos with last-token mean-diff"])
directions
""")

md("""
The whitened direction is essentially orthogonal to the canonical mean-diff direction in 5376-dim space (cos = 0.0155). All three achieve similar probe AUC. There is no canonical refusal direction — it depends on extraction method, pooling, and layer.
""")

code("""
def patch_summary(label, blob):
    a = blob["aggregate"]
    return {
        "direction":         label,
        "n":                 a["n"],
        "baseline":          f"{a['baseline_compliance']}/{a['n']}",
        "ablated (judge)":   f"{a['ablated_compliance']}/{a['n']}",
        "random (judge)":    f"{a['random_compliance']}/{a['n']}",
        "ablated rate":      a["ablated_rate"],
    }

dir_compare = pd.DataFrame([
    patch_summary("last-token, all-15 layers",    patch_all15),
    patch_summary("whitened SVD, all layers",     patch_whit),
    patch_summary("mean-pool multi-layer",        patch_mp),
])
dir_compare
""")

md("""
Whitened-direction ablation barely flips anything (1/20 ablated, indistinguishable from baseline 1/20) and the random control flips 7/20 — whitening removes too much, the resulting direction is not the causal one. Multi-layer mean-pool ablation shows 8/20 judge-flagged but is pending manual audit; given random control 9/20 the signal is unclear without the audit.
""")

# ============================================================
# Section 6 — Cyber-cascade
# ============================================================
md("""
## Section 6 — Stretch goal: cyber-cascade flip

We ran PRE on n=30 cyber-3 (prohibited) prompts using the cyber-3 probe direction as the projection scorer. Methodology caveat: `pre_agent.py`'s behavior judge classifies refusal vs compliance, not "is this still asking for cyber-prohibited content". So this measures Gemma's compliance with cyber-3-flagged rewrites — a useful proxy for the strict cascade-flip definition (all three cyber probes below threshold simultaneously) but not identical to it.
""")

code("""
a = pre_cy3["aggregate"]
cy = pd.DataFrame([
    ("n",                     a["n_total"]),
    ("n intent-preserved",    a["n_intent_preserved"]),
    ("Pr(f|edit) judge",      f"{a['Pr_f_given_edit']:.3f}"),
    ("Pr(model|edit) judge",  f"{a['Pr_model_given_edit']:.3f}"),
    ("mean Δprobe",           f"{a['mean_delta_probe']:+.3f}"),
], columns=["", "value"]).set_index("")
cy
""")

md("""
Manual audit of 22 judge-flagged compliances: 13 true flips (reverse shells, RATs, exfiltration, XSS payloads), 4 defensive-but-dual-use, 5 safe/partial. Conservative true Pr(model|edit) = 13/27 = 0.481 [0.302, 0.665]. Mean Δprobe = +0.08 — the probe score actually went up on average. The cyber-3 probe direction is essentially uncorrelated with what controls Gemma's compliance on cyber-prohibited prompts.
""")

# ============================================================
# Section 7 — Headline summary
# ============================================================
md("""
## Section 7 — Headline

Four manually-audited interventions on the same probe direction, with Wilson 95% CIs.
""")

code("""
headline_rows = [
    ("Token edits (minimal, in-spec)",          20, 0,   wilson(0,  20)),
    ("PRE K=7 (paraphrase, in-spec)",           66, 39,  wilson(39, 66)),
    ("Activation ablation L48 (diagnostic)",    20, 16,  wilson(16, 20)),
    ("Cyber-cascade stretch (PRE on cyber-3)",  27, 13,  wilson(13, 27)),
]
headline = pd.DataFrame([
    {"intervention": label,
     "n": n,
     "true flip": f"{k}/{n}",
     "rate": k / n,
     "Wilson 95%": f"[{ci[0]:.3f}, {ci[1]:.3f}]"}
    for (label, n, k, ci) in headline_rows
])
headline
""")

code("""
fig, ax = plt.subplots(figsize=(9, 4.2))
labels = ["Token edits", "PRE K=7", "Ablation L48", "Cyber-cascade"]
rates  = [k/n for (_, n, k, _) in headline_rows]
los    = [ci[0] for (_, _, _, ci) in headline_rows]
his    = [ci[1] for (_, _, _, ci) in headline_rows]
err_lo = [r - lo for r, lo in zip(rates, los)]
err_hi = [hi - r for r, hi in zip(rates, his)]

bars = ax.bar(labels, rates, yerr=[err_lo, err_hi], capsize=6,
              color=["#7f8c8d", "#27ae60", "#c0392b", "#8e44ad"],
              alpha=0.85, edgecolor="black", linewidth=0.5)
for bar, (label, n, k, _) in zip(bars, headline_rows):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{k}/{n}", ha="center", fontsize=9)

ax.set_ylabel("Manual TRUE compliance rate")
ax.set_ylim(0, 1.05)
ax.set_title("Four interventions, manually audited (Wilson 95% CIs)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
""")

md("""
The probe direction is a causal handle, accessible via paraphrase from input space (~60%) and via single-layer residual ablation at L48 (~80%). It is not accessible via token-level attribution: the attribution map points at structural correlates (newlines, indentation, BPE fragments), not features the model uses to decide. AUC against a static eval set does not predict robustness against paraphrase attack — Constitutional Classifiers as production probe gates inherit this gap.

The single-direction abstraction is layer-specific and extraction-specific. Whitening and mean-pooling produce nearly orthogonal axes that all achieve similar AUC; only last-token mean-diff at L48 gives a working ablation. "The refusal direction" is a useful shorthand, not a unique object.
""")

# ============================================================
# Section 8 — Reproducibility
# ============================================================
md("""
## Section 8 — Reproducibility

Each result corresponds to a script in the repo root. All JSON outputs in `edit_eval/` are checkpointed per-prompt, so partial runs resume cleanly. Wilson CIs come from the helper in `pre_agent.py` (mirrored above).
""")

code("""
scripts = pd.DataFrame([
    ("edit_agent.py",  "level2_final.json",            "n=20 token edits, attribute → edit → judge → verify"),
    ("pre_agent.py",   "pre_n81.json / pre_k{7,10}_n81.json",
                                                       "n=81 free-form rewrites, K∈{3,7,10} paraphrases per prompt"),
    ("pre_agent.py",   "pre_whitened_n81.json",        "PRE with whitened-SVD direction"),
    ("pre_agent.py",   "pre_cyber3_cascade.json",      "PRE on n=30 cyber-3 prompts, cyber-3 probe as scorer"),
    ("patch_agent.py", "patch_full.json",              "all-15-layer ablation (over-specifies, washes out)"),
    ("patch_agent.py", "patch_layer_L*.json",          "single-layer ablation sweep, L4..L56 every 4 layers"),
    ("patch_agent.py", "patch_whitened_n20.json",      "ablation with whitened-SVD direction"),
    ("patch_agent.py", "patch_multilayer_meanpool_n20.json",
                                                       "ablation with mean-pool multi-layer direction"),
    ("build_dirs.py",  "/scratch/hybrid_v2/...",       "recompute mean-diff directions (last-token / mean-pool / whitened)"),
], columns=["script", "produces", "notes"])
scripts
""")

# ============================================================
# Build + write
# ============================================================
nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
})

OUT.write_text(nbf.writes(nb))
print(f"Wrote {OUT}  ({len(cells)} cells)")

# Execute in repo root so relative `edit_eval/` resolves.
print("Executing notebook...")
client = NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(REPO)}})
client.execute()
OUT.write_text(nbf.writes(nb))
print("Executed and saved.")
