"""Extract single-layer directions from the 16-layer file built by
the hybrid pipeline.

For the layer-sweep experiment we need to ablate at ONE layer at a time.
patch_agent installs a hook for every layer in `layer_idxs`, so by
producing single-layer direction files we get single-layer ablations.

Outputs to /scratch/hybrid_v2/gemma4_31b/direction_single_L{N}.pt for
each N in TARGET_LAYERS.
"""
import torch
from pathlib import Path

SRC = "/scratch/hybrid/gemma4_31b/direction_refusal_gemma4_31b.pt"
OUT_DIR = Path("/scratch/hybrid_v2/gemma4_31b")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_LAYERS = [16, 32, 44, 48, 56]


def main():
    src = torch.load(SRC, weights_only=False, map_location="cpu")
    all_layers = src["layer_idxs"]
    print(f"[split] source has layers {all_layers}", flush=True)

    for L in TARGET_LAYERS:
        if L not in all_layers:
            print(
                f"[split] WARNING: layer {L} not in source, skipping",
                flush=True,
            )
            continue
        idx = all_layers.index(L)
        direction_at_L = src["direction"][idx]   # [5376]
        norm_at_L = src["norms"][idx]
        raw = direction_at_L.norm().item()
        if raw < 0.5:
            print(
                f"[split] layer {L} direction is degenerate "
                f"(norm={raw:.4f}), skipping",
                flush=True,
            )
            continue
        norm_val = (norm_at_L.item()
                    if hasattr(norm_at_L, "item") else float(norm_at_L))
        out = {
            "direction": direction_at_L.unsqueeze(0).float(),
            "norms": torch.tensor([norm_val]).float(),
            "layer_idxs": [L],
            "task": f"refusal_single_L{L}",
        }
        path = OUT_DIR / f"direction_single_L{L}.pt"
        torch.save(out, path)
        print(f"[split] saved L{L} -> {path}", flush=True)


if __name__ == "__main__":
    main()
