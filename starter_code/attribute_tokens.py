"""Level 2 — Per-token attribution using refusal direction loading.

For each sample, compute per-token refusal-direction loading weighted by
layer importance (from XGBoost feature importance). This identifies which
tokens in the prompt drive the model toward refusal.

Outputs a JSON compatible with iterative_edit_agent.py's eval_set format.

Usage:
    python attribute_tokens.py \
        --model_key gemma4_31b \
        --artifacts_dir ./hybrid_artifacts/gemma4_31b \
        --models_dir ./hybrid_models/gemma4_31b \
        --out_dir ./edit_eval
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from extract_residuals import resolve_model_path, parse_layers, HF_REPOS
from chunked_sdpa import chunked_sdpa_scope

DEVICE = "cuda:0"


def load_directions_and_weights(artifacts_dir, models_dir, task_name):
    """Load direction + layer importance weights from XGBoost."""
    import xgboost as xgb

    # Direction
    d = torch.load(str(artifacts_dir / f"direction_{task_name}.pt"),
                   weights_only=False, map_location="cpu")
    direction = d["direction"]  # (n_layers_sel, d_model)
    layer_idxs = d["layer_idxs"]

    # XGBoost feature importance → layer weights
    booster = xgb.Booster()
    booster.load_model(str(models_dir / f"xgb_{task_name}.json"))
    importance = booster.get_score(importance_type="gain")

    # dir_loading features are f0..f{n_layers-1}
    n_layers = direction.shape[0]
    layer_weights = np.zeros(n_layers)
    for i in range(n_layers):
        key = f"f{i}"
        if key in importance:
            layer_weights[i] = importance[key]
    # Normalize
    total = layer_weights.sum()
    if total > 0:
        layer_weights /= total
    else:
        layer_weights[:] = 1.0 / n_layers

    return direction, torch.from_numpy(layer_weights).float(), layer_idxs


def attribute_tokens(residuals, direction, layer_weights):
    """Per-token attribution via weighted refusal-direction loading.

    Args:
        residuals: (n_layers, n_tokens, d_model) float
        direction: (n_layers, d_model) unit vectors
        layer_weights: (n_layers,) from XGBoost importance

    Returns:
        (n_tokens,) attribution scores
    """
    # Per-token, per-layer loading
    # (n_layers, n_tokens) = einsum("l t d, l d -> l t")
    loadings = torch.einsum("ltd,ld->lt", residuals, direction)

    # Weight by layer importance
    # (n_tokens,) = einsum("l t, l -> t")
    weighted = torch.einsum("lt,l->t", loadings, layer_weights)

    return weighted.numpy()


def get_functional_ids(tok):
    """Get set of special/functional token IDs to exclude from edits."""
    ids = set(tok.all_special_ids or [])
    if hasattr(tok, "added_tokens_encoder"):
        ids |= set(tok.added_tokens_encoder.values())
    for s in ["<bos>", "<eos>", "<|turn>", "<turn|>", "<start_of_turn>",
              "<end_of_turn>", "<pad>", "<unk>", "<|im_start|>",
              "<|im_end|>", "<|endoftext|>"]:
        try:
            tid = tok.convert_tokens_to_ids(s)
            if tid is not None and tid != tok.unk_token_id:
                ids.add(int(tid))
        except Exception:
            pass
    return ids


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_key", default="gemma4_31b",
                    choices=list(HF_REPOS.keys()))
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--artifacts_dir", required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--task", default=None,
                    help="Task name (default: refusal_<model_key>)")
    ap.add_argument("--out_dir", default="./edit_eval")
    ap.add_argument("--limit_tokens", type=int, default=2048,
                    help="Skip prompts longer than this")
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--sample_limit", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    task_name = args.task or f"refusal_{args.model_key}"
    artifacts_dir = Path(args.artifacts_dir)
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Token Attribution: {task_name} ===", flush=True)

    # Load directions + weights
    direction, layer_weights, layer_idxs = load_directions_and_weights(
        artifacts_dir, models_dir, task_name)
    n_layers_sel = direction.shape[0]
    print(f"Direction: {direction.shape}, "
          f"Layer weights: {layer_weights.tolist()[:5]}...", flush=True)

    # Load model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = resolve_model_path(args.model_key, args.model_path)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=DEVICE, trust_remote_code=True)
    model.eval()
    n_layers = len(model.model.layers) if hasattr(model.model, "layers") else 64
    functional_ids = get_functional_ids(tok)

    # Load refusal samples (test split, refusals only)
    base = Path(__file__).resolve().parent.parent / "datasets"
    if "refusal" in task_name:
        samples = [json.loads(l) for l in open(
            base / "refusal_probes" / args.model_key / "test_split.jsonl")]
        # Only refusals (positive class) — these are what we want to flip
        samples = [s for s in samples if s["is_refusal"]]
        prompt_key = "attack_prompt"
    else:
        samples = [json.loads(l) for l in open(
            base / "cyber_probes" / "test.jsonl")]
        pdef = {"cyber_probe3": "prohibited", "cyber_probe2": "high_risk_dual_use",
                "cyber_probe1": "dual_use"}
        cat = pdef.get(task_name, "prohibited")
        samples = [s for s in samples if s["category"] == cat]
        prompt_key = "prompt"

    # Filter by token length
    samples = [s for s in samples if s.get("n_tokens", 0) <= args.limit_tokens]
    if args.sample_limit > 0:
        samples = samples[:args.sample_limit]
    print(f"Samples for attribution: {len(samples)}", flush=True)

    # Forward pass + attribution
    use_chunked = (args.model_key == "gemma4_31b")
    cm = chunked_sdpa_scope() if use_chunked else None
    if cm is not None:
        cm.__enter__()

    records = []
    t_start = time.time()
    try:
        for i, s in enumerate(samples):
            sid = s["sample_id"]
            prompt = s[prompt_key]
            txt = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
            enc = tok(txt, return_tensors="pt").to(DEVICE)
            ids_t, attn = enc.input_ids, enc.attention_mask

            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model(input_ids=ids_t, attention_mask=attn,
                            output_hidden_states=True, return_dict=True)

            hs = out.hidden_states
            n_tok = ids_t.shape[1]

            # Stack selected layers: (n_layers_sel, n_tok, d_model)
            residuals = torch.stack(
                [hs[k][0] for k in layer_idxs], dim=0
            ).cpu().float()

            del out, hs
            torch.cuda.empty_cache()

            # Attribution
            attrib = attribute_tokens(residuals, direction, layer_weights)

            # Decode tokens
            ids_list = ids_t[0].cpu().tolist()
            pieces = tok.convert_ids_to_tokens(ids_list)
            pieces = [p.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")
                       for p in pieces]

            # Top-K editable tokens
            editable = np.array([
                int(ids_list[j]) not in functional_ids
                for j in range(n_tok)])
            scores = np.where(editable, attrib[:n_tok], -np.inf)
            top_indices = np.argsort(-scores)[:args.topk]

            top_tokens = [
                {"position": int(j), "token_text": pieces[j] if j < len(pieces) else "",
                 "score": float(attrib[j])}
                for j in top_indices if scores[j] > -np.inf
            ]

            records.append({
                "sample_id": sid,
                "method": "hybrid_refusal_dir",
                "model_key": args.model_key,
                "task": task_name,
                "n_tokens": n_tok,
                "prompt_text_decoded": tok.decode(ids_list, skip_special_tokens=False),
                "top_k_tokens": top_tokens,
                "all_token_attrib": [round(float(x), 5) for x in attrib[:n_tok]],
                "all_token_text": pieces[:n_tok],
            })

            if (i + 1) % 10 == 0 or (i + 1) == len(samples):
                print(f"  [{i+1}/{len(samples)}] {sid}: n_tok={n_tok} "
                      f"top1={top_tokens[0]['token_text'][:20]!r} "
                      f"score={top_tokens[0]['score']:.3f}", flush=True)
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)

    out_path = out_dir / f"attribution_{task_name}_hybrid.json"
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"\nDONE: {out_path} ({len(records)} records, "
          f"{(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
