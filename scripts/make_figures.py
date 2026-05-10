"""Generate the 4 figures for the slide deck."""
import json
import glob
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# UPDATE THESE post-audit. Manual TRUE flip counts.
MANUAL = {
    "token_edits":         {"true_flip": 0,  "n": 20},
    "pre_k3":              {"true_flip": 25, "n": 64},
    "pre_k7":              {"true_flip": 39, "n": 66},
    "pre_k10":             {"true_flip": 38, "n": 61},
    "ablation_all15":      {"true_flip": 0,  "n": 20},
    "ablation_L48":        {"true_flip": 16, "n": 20},
    "ablation_random_L48": {"true_flip": 0,  "n": 20},
    "cyber_cascade":       {"true_flip": 13, "n": 27},
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bar_with_ci(ax, x, p, lo, hi, k, n, color="#3b6dad"):
    ax.bar(x, p, width=0.55, color=color, edgecolor="black", linewidth=0.8)
    ax.errorbar(x, p, yerr=[[p - lo], [hi - p]],
                fmt="none", color="black", capsize=8, linewidth=1.4)
    ax.text(x, hi + 0.03, f"{k}/{n}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")


def fig_three_bar(out_path):
    bars = [
        ("Token edits\n(probe attribution)", MANUAL["token_edits"], "#7f8c8d"),
        ("Free-form rewrite\n(PRE, K=7)", MANUAL["pre_k7"], "#3498db"),
        ("Activation ablation\n(L48 single-layer)", MANUAL["ablation_L48"], "#e74c3c"),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, m, color) in enumerate(bars):
        k, n = m["true_flip"], m["n"]
        p = k / n
        lo, hi = wilson(k, n)
        bar_with_ci(ax, i, p, lo, hi, k, n, color=color)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=11)
    ax.set_ylabel("Pr(model flipped | intervention)\n(manually verified)", fontsize=12)
    ax.set_title("Three interventions on the same probe direction\n"
                 "Gemma-4-31B-it, probe AUC 0.929", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")


def fig_layer_sweep(out_path, sweep_dir="edit_eval"):
    files = sorted(glob.glob(f"{sweep_dir}/patch_layer_L*.json"),
                   key=lambda x: int(x.split("_L")[-1].split(".")[0]))
    layers, ablated, baseline, random_ = [], [], [], []
    for f in files:
        try:
            d = json.load(open(f))
            agg = d["aggregate"]
            n_results = len(d.get("results", []))
            if n_results < 10:
                continue
            L = int(f.split("_L")[-1].split(".")[0])
            layers.append(L)
            ablated.append(agg["ablated_compliance"] / agg["n"])
            baseline.append(agg["baseline_compliance"] / agg["n"])
            random_.append(agg["random_compliance"] / agg["n"])
        except Exception as e:
            print(f"skipping {f}: {e}")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(layers, ablated, "o-", color="#c0392b", linewidth=2.4, markersize=9,
            label="Ablated (refusal direction)")
    ax.plot(layers, random_, "s--", color="#7f7f7f", linewidth=1.6, markersize=7,
            label="Random direction (control)")
    ax.plot(layers, baseline, "^:", color="#2c3e50", linewidth=1.4, markersize=6,
            label="Baseline (no intervention)")
    ax.set_xlabel("Ablation layer (single-layer intervention)", fontsize=12)
    ax.set_ylabel("Compliance rate (judge-reported)", fontsize=12)
    ax.set_title("Layer-sweep ablation: where the refusal direction is causal\n"
                 "Gemma-4-31B-it, n=20 per layer", fontsize=13)
    ax.set_xticks(layers)
    ax.set_ylim(-0.02, 1.0)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if 48 in layers:
        idx = layers.index(48)
        ax.annotate(f"peak L48\n{ablated[idx]:.0%} judge", xy=(48, ablated[idx]),
                    xytext=(52, ablated[idx] + 0.04), fontsize=11, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#c0392b"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")


def fig_k_sweep(out_path):
    Ks = [3, 7, 10]
    keys = ["pre_k3", "pre_k7", "pre_k10"]
    rates, los, his, counts = [], [], [], []
    for k in keys:
        m = MANUAL[k]
        p = m["true_flip"] / m["n"]
        lo, hi = wilson(m["true_flip"], m["n"])
        rates.append(p); los.append(lo); his.append(hi); counts.append(m)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(Ks, rates,
                yerr=[[r - l for r, l in zip(rates, los)],
                      [h - r for r, h in zip(rates, his)]],
                fmt="o-", color="#1f77b4", linewidth=2.5, markersize=11,
                capsize=8)
    for K, r, m in zip(Ks, rates, counts):
        ax.text(K, r + 0.04, f"{m['true_flip']}/{m['n']}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("K (paraphrase candidates per prompt)", fontsize=12)
    ax.set_ylabel("Pr(model flipped | edit), manual TRUE", fontsize=12)
    ax.set_title("PRE: paraphrase budget vs flip rate (n=81 per K)", fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(Ks)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")


def fig_delta_vs_flip(out_path, pre_file="edit_eval/pre_k7_n81.json"):
    if not Path(pre_file).exists():
        print(f"skipping {out_path}: {pre_file} not found")
        return
    d = json.load(open(pre_file))
    results = d.get("results", [])
    flipped_x, flipped_y = [], []
    refused_x, refused_y = [], []
    np.random.seed(0)
    for r in results:
        dp = r.get("delta_probe")
        if dp is None:
            continue
        if r.get("edited_behavior") == "compliance":
            flipped_x.append(dp); flipped_y.append(1 + np.random.uniform(-0.04, 0.04))
        elif r.get("edited_behavior") == "refusal":
            refused_x.append(dp); refused_y.append(0 + np.random.uniform(-0.04, 0.04))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(flipped_x, flipped_y, s=70, alpha=0.7,
               color="#c0392b", edgecolor="black", linewidth=0.6,
               label=f"flipped (n={len(flipped_x)})")
    ax.scatter(refused_x, refused_y, s=70, alpha=0.7,
               color="#3498db", edgecolor="black", linewidth=0.6,
               label=f"refused (n={len(refused_x)})")
    ax.axvline(0, linestyle="--", color="black", alpha=0.5, linewidth=1)
    n_off_axis = sum(1 for x in flipped_x if x >= 0)
    ax.text(0.02, 1.15, f"{n_off_axis} off-axis flips →", fontsize=10, color="#c0392b")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["refusal", "compliance"], fontsize=11)
    ax.set_xlabel("Δprobe = score(rewrite) − score(original)", fontsize=12)
    ax.set_title("Behavior flip vs probe movement (PRE K=7, n=66 intent-preserved)\n"
                 "Compliance with Δprobe ≥ 0 = probe missed it",
                 fontsize=12)
    ax.legend(loc="center right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.3, 1.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")


def main():
    out = Path("/scratch/mechhack/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig_three_bar(out / "fig_three_bar.png")
    fig_layer_sweep(out / "fig_layer_sweep.png")
    fig_k_sweep(out / "fig_k_sweep.png")
    fig_delta_vs_flip(out / "fig_delta_vs_flip.png")
    print(f"\nAll 4 figures in {out}")


if __name__ == "__main__":
    main()
