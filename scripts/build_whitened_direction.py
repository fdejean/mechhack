"""Compute whitened-SVD refusal direction at layer 48 from cached residuals.

Math:
    d_meandiff   = mean(act | refusal=1) - mean(act | refusal=0)
    Sigma        = pooled within-class covariance of all activations
    d_whitened   = (Sigma + ridge*I)^{-1} @ d_meandiff
    return d_whitened / ||d_whitened||

Theory: mean-diff captures discriminative direction in raw activation space.
Whitening removes contamination by high-natural-variance directions, leaving
the direction that maximally separates classes per unit within-class variance.
This is Mahalanobis-style direction extraction; OBLITERATUS uses it because
it produces cleaner refusal directions than naive mean-diff on Llama-3.

Outputs to /scratch/hybrid_v2/gemma4_31b/direction_arditi_whitened_L48.pt
in the format patch_agent expects: {direction, norms, layer_idxs, task}.
"""
import torch
import glob
import time
from pathlib import Path

EXTRACT_DIR = "/scratch/mechhack/extracts/gemma4_31b_multilayer/gemma4_31b"
LAYER_IDXS = [8, 16, 32, 48, 56]
LAYER_TARGET = 48
LAYER_POS = LAYER_IDXS.index(LAYER_TARGET)
OUT_DIR = Path("/scratch/hybrid_v2/gemma4_31b")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RIDGE = 1e-3   # regularization for covariance inversion


def main():
    print(f"[whiten] reading from {EXTRACT_DIR}", flush=True)
    files = sorted(glob.glob(f"{EXTRACT_DIR}/*.pt"))
    print(f"[whiten] {len(files)} files", flush=True)

    t0 = time.time()
    acts_pos, acts_neg = [], []
    for i, f in enumerate(files):
        try:
            d = torch.load(f, weights_only=False, map_location="cpu")
        except Exception:
            continue
        res = d["residuals"][LAYER_POS]   # [n_tok, 5376] fp16
        n_tok = int(d["n_tokens"])
        last = res[n_tok - 1].float()
        if d["label"] == 1:
            acts_pos.append(last)
        else:
            acts_neg.append(last)
        if (i + 1) % 100 == 0:
            print(
                f"  {i+1}/{len(files)} | "
                f"n_pos={len(acts_pos)} n_neg={len(acts_neg)} | "
                f"{time.time()-t0:.1f}s",
                flush=True,
            )

    acts_pos = torch.stack(acts_pos)   # [N_pos, 5376]
    acts_neg = torch.stack(acts_neg)   # [N_neg, 5376]
    print(
        f"[whiten] pos={acts_pos.shape}, neg={acts_neg.shape}",
        flush=True,
    )

    # 1. Mean-difference direction
    mu_pos = acts_pos.mean(0)
    mu_neg = acts_neg.mean(0)
    d_meandiff = mu_pos - mu_neg
    print(
        f"[whiten] meandiff norm = {d_meandiff.norm().item():.4f}",
        flush=True,
    )

    # 2. Pooled within-class covariance
    acts_all = torch.cat(
        [acts_pos - mu_pos, acts_neg - mu_neg],
        dim=0,
    )
    N = acts_all.shape[0]
    print(
        f"[whiten] computing covariance over {N} centered samples...",
        flush=True,
    )
    Sigma = (acts_all.T @ acts_all) / (N - 1)   # [5376, 5376]
    print(
        f"[whiten] Sigma shape={Sigma.shape}, "
        f"mean diag={Sigma.diag().mean().item():.4f}",
        flush=True,
    )

    # 3. Whitened direction via regularized solve (avoids 5376x5376 pinv)
    print(
        f"[whiten] solving (Sigma + {RIDGE}*I) x = meandiff...",
        flush=True,
    )
    Sigma_reg = Sigma + RIDGE * torch.eye(Sigma.shape[0])
    d_whitened = torch.linalg.solve(
        Sigma_reg,
        d_meandiff.unsqueeze(-1),
    ).squeeze(-1)
    raw_norm = d_whitened.norm().item()
    print(
        f"[whiten] whitened norm (unnormalized) = {raw_norm:.4f}",
        flush=True,
    )

    # 4. Normalize
    d_whitened_unit = d_whitened / d_whitened.norm()
    d_meandiff_unit = d_meandiff / d_meandiff.norm()
    cos_sim = torch.dot(d_whitened_unit, d_meandiff_unit).item()
    print(
        f"[whiten] cos(whitened, meandiff) = {cos_sim:.4f}",
        flush=True,
    )

    # 5. Save in patch_agent's expected format
    out = {
        "direction": d_whitened_unit.unsqueeze(0).float(),   # [1, 5376]
        "norms": torch.tensor([raw_norm]).float(),
        "layer_idxs": [LAYER_TARGET],
        "task": f"refusal_arditi_whitened_L{LAYER_TARGET}",
        "ridge": RIDGE,
        "cos_with_meandiff": cos_sim,
    }
    out_path = OUT_DIR / f"direction_arditi_whitened_L{LAYER_TARGET}.pt"
    torch.save(out, out_path)
    print(f"[whiten] saved to {out_path}", flush=True)
    print(f"[whiten] total: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
