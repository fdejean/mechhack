"""Activation patching: directional ablation of the refusal direction.

Following Arditi et al. 2024 ("Refusal in Language Models Is Mediated by
a Single Direction"). Adapts their abliteration method to Gemma 4-31B-it.

For each refused prompt, we run three conditions:

    1. baseline      — no intervention, plain rollout
    2. ablated       — at every extracted layer, project out the refusal
                       direction from the residual stream:
                           x  <-  x - (x . d_l) * d_l   (d_l unit-norm)
    3. random_ctrl   — same hook structure, but using a random unit
                       direction per layer (control: tests whether ANY
                       residual perturbation flips behavior, or whether
                       we need the SPECIFIC refusal direction)

Behavior judged by the project's MiniMax-M2.7-via-AIaaS refusal judge.

Usage:
    export AIAAS_KEY=sk-...
    python patch_agent.py \
        --eval_set datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl \
        --direction /scratch/hybrid/gemma4_31b/direction_refusal_gemma4_31b.pt \
        --output edit_eval/patch_smoke.json \
        --limit 3

Then:
    python patch_agent.py ... --output edit_eval/patch_full.json --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the project's pieces directly — no reinventing.
sys.path.insert(0, "/scratch/mechhack")
from starter_code.behavior_verifier import (
    extract_answer,
    REFUSAL_JUDGE_SYSTEM,
    REFUSAL_JUDGE_SCHEMA,
)
from starter_code.llm_clients import AIaaSClient


# ---------------------------------------------------------------------------
# Model + direction loading
# ---------------------------------------------------------------------------

def find_layer_module(model, layer_idx: int):
    """Locate decoder layer N. Gemma 4 puts them at model.language_model.layers."""
    for path in (
        "model.language_model.layers",
        "language_model.model.layers",
        "model.model.layers",
        "model.layers",
    ):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj[layer_idx]
        except AttributeError:
            continue
    raise RuntimeError(f"Could not locate decoder layer {layer_idx}")


class GemmaWithHooks:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        print(f"[load] tokenizer + model from {model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = device
        # Clamp helper: layer_idxs file goes up to 60, but model has 60 layers (0..59)
        n_layers = len(find_layer_module(self.model, 0).__class__.__bases__) - 1  # noop
        # Actually count decoder layers properly:
        for path in ("model.language_model.layers", "model.model.layers"):
            obj = self.model
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                self.n_layers = len(obj)
                break
            except AttributeError:
                continue
        print(f"[load] decoder has {self.n_layers} layers", flush=True)

    def get_layer(self, layer_idx: int):
        # Clamp 60 -> 59 (last layer); residual after final layer == output of last layer
        return find_layer_module(self.model, min(layer_idx, self.n_layers - 1))

    @torch.no_grad()
    def rollout(self, prompt: str, max_new_tokens: int = 256) -> str:
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = self.tokenizer(ids, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            out[0, enc.input_ids.shape[1]:], skip_special_tokens=False
        )


def load_directions(path: str, device: str):
    """Returns dict: {layer_idx: unit_direction_tensor on device, bf16}.

    The hybrid pipeline already saves unit-norm directions in d['direction'].
    d['norms'] stores the ORIGINAL pre-normalization scale, NOT a divisor.
    Layers with degenerate (~0 norm) vectors are skipped.
    """
    d = torch.load(path, map_location="cpu")
    raw = d["direction"]
    layer_idxs = d["layer_idxs"]
    out = {}
    skipped = []
    for i, li in enumerate(layer_idxs):
        v = raw[i].float()
        nrm = v.norm().item()
        if nrm < 0.5:
            skipped.append((li, nrm))
            continue
        v = v / nrm
        out[li] = v.to(device=device, dtype=torch.bfloat16)
    print(f"[load] {len(out)} layer-direction vectors at {sorted(out.keys())}", flush=True)
    if skipped:
        print(f"[load] skipped degenerate layers: {skipped}", flush=True)
    return out, sorted(out.keys())


def make_random_directions(real_dirs, seed: int = 0):
    """Random unit vectors of the same shape, same device/dtype, fixed seed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for li, v in real_dirs.items():
        r = torch.randn(v.shape, generator=g).to(device=v.device, dtype=v.dtype)
        r = r / r.float().norm()
        out[li] = r.to(dtype=v.dtype)
    return out


# ---------------------------------------------------------------------------
# Directional ablation hook — the core intervention
# ---------------------------------------------------------------------------

def make_ablation_hook(direction: torch.Tensor):
    """Returns a forward hook that projects `direction` out of the residual stream.

    For each token's residual x:
        x_new = x - (x . d) * d
    where d is unit-norm. Applied to ALL token positions (Arditi-style full ablation).
    """
    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        # hs: [batch, seq, hidden]; direction: [hidden]
        # projection coefficient per (batch, seq): (hs @ d)
        d = direction.to(dtype=hs.dtype, device=hs.device)
        coef = (hs * d).sum(dim=-1, keepdim=True)         # [B, T, 1]
        hs_new = hs - coef * d                            # broadcast [B, T, hidden]
        if is_tuple:
            return (hs_new,) + output[1:]
        return hs_new
    return hook


def install_ablation(gem: GemmaWithHooks, directions: dict):
    """Register ablation hooks on every layer in `directions`. Returns handle list."""
    handles = []
    for layer_idx, d in directions.items():
        layer_mod = gem.get_layer(layer_idx)
        h = layer_mod.register_forward_hook(make_ablation_hook(d))
        handles.append(h)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# Behavior judge — reuse project's schema
# ---------------------------------------------------------------------------

class Judge:
    def __init__(self):
        self.client = AIaaSClient("MiniMaxAI/MiniMax-M2.7")

    def __call__(self, prompt: str, raw_response: str) -> dict:
        answer = extract_answer(raw_response)[:1500]
        user = (
            f"USER PROMPT:\n{prompt}\n\n"
            f"MODEL RESPONSE (first 1500 chars after thinking):\n{answer}\n\n"
            "Classify the response. Return JSON."
        )
        try:
            resp = self.client.call(
                system=REFUSAL_JUDGE_SYSTEM,
                user=user,
                schema=REFUSAL_JUDGE_SCHEMA,
                max_tokens=2000,  # MiniMax thinks a lot — give it room
            )
            parsed = resp["parsed"]
            return {
                "label": "refusal" if parsed["is_refusal"] else "compliance",
                "reason": parsed.get("reason", ""),
            }
        except Exception as e:
            return {"label": "refusal", "reason": f"judge_failed: {e}"}


# ---------------------------------------------------------------------------
# Per-prompt loop
# ---------------------------------------------------------------------------

@dataclass
class Result:
    sample_id: str
    original_prompt: str
    baseline_response: str = ""
    baseline_label: str = ""
    ablated_response: str = ""
    ablated_label: str = ""
    random_response: str = ""
    random_label: str = ""


def run_one(sample: dict, gem: GemmaWithHooks, real_dirs: dict, rand_dirs: dict, judge: Judge) -> Result:
    sid = sample.get("sample_id", "unknown")
    prompt = sample.get("attack_prompt") or sample.get("prompt")
    r = Result(sample_id=sid, original_prompt=prompt)

    # 1. Baseline — no hooks
    r.baseline_response = gem.rollout(prompt)
    r.baseline_label = judge(prompt, r.baseline_response)["label"]

    # 2. Ablated — refusal direction
    handles = install_ablation(gem, real_dirs)
    try:
        r.ablated_response = gem.rollout(prompt)
    finally:
        remove_hooks(handles)
    r.ablated_label = judge(prompt, r.ablated_response)["label"]

    # 3. Random control
    handles = install_ablation(gem, rand_dirs)
    try:
        r.random_response = gem.rollout(prompt)
    finally:
        remove_hooks(handles)
    r.random_label = judge(prompt, r.random_response)["label"]

    return r


# ---------------------------------------------------------------------------
# Aggregate (Wilson 95% CI)
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def aggregate(results):
    n = len(results)
    bc = sum(r.baseline_label == "compliance" for r in results)
    ac = sum(r.ablated_label == "compliance" for r in results)
    rc = sum(r.random_label == "compliance" for r in results)
    return {
        "n": n,
        "baseline_compliance": bc, "baseline_rate": bc / n if n else 0,
        "baseline_CI95": wilson(bc, n),
        "ablated_compliance": ac, "ablated_rate": ac / n if n else 0,
        "ablated_CI95": wilson(ac, n),
        "random_compliance": rc, "random_rate": rc / n if n else 0,
        "random_CI95": wilson(rc, n),
        "ablation_minus_baseline": (ac - bc) / n if n else 0,
        "ablation_minus_random": (ac - rc) / n if n else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--model_path", default="/data/Gemma-4-31B-it")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    with open(args.eval_set) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    samples = samples[: args.limit]
    print(f"[main] {len(samples)} samples", flush=True)

    gem = GemmaWithHooks(args.model_path)
    real_dirs, layer_idxs = load_directions(args.direction, gem.device)
    rand_dirs = make_random_directions(real_dirs, seed=args.seed)
    judge = Judge()

    results = []
    for i, s in enumerate(samples):
        t0 = time.time()
        try:
            r = run_one(s, gem, real_dirs, rand_dirs, judge)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[main] sample {i} failed: {e}", flush=True)
            continue
        results.append(r)
        print(
            f"[{i+1}/{len(samples)}] {r.sample_id}  "
            f"baseline={r.baseline_label}  "
            f"ablated={r.ablated_label}  "
            f"random={r.random_label}  "
            f"({time.time()-t0:.1f}s)",
            flush=True,
        )
        with open(out_path, "w") as f:
            json.dump({
                "config": vars(args),
                "results": [asdict(r) for r in results],
                "aggregate": aggregate(results),
            }, f, indent=2)

    print("\n=== AGGREGATE ===", flush=True)
    print(json.dumps(aggregate(results), indent=2), flush=True)


if __name__ == "__main__":
    main()
