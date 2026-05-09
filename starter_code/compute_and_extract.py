"""Hybrid pipeline: compute refusal/harm directions + extract features.

Fused single-model-load approach:
  1. Forward all unique samples → cache last-token residuals per layer
  2. Compute class-conditional mean-difference directions from train split
  3. Compute per-sample feature vectors (direction loadings + logit lens)
  4. Save features (.npz), directions (.pt), and W_U (.pt)

Usage on the pod:
    python compute_and_extract.py \
        --model_key gemma4_31b \
        --layers "0:62:4" \
        --out_dir /scratch/hybrid

    # Qwen (after Gemma finishes):
    python compute_and_extract.py \
        --model_key qwen36 \
        --layers "0:66:4" \
        --out_dir /scratch/hybrid
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from chunked_sdpa import chunked_sdpa_scope
from extract_residuals import resolve_model_path, parse_layers, HF_REPOS

DEVICE = "cuda:0"

# ---------- token sets for logit lens ----------
REFUSAL_STARTERS = [
    "I", "Sorry", "cannot", "unable", "Unfortunately", "apologize",
    "can't", "won't", "No", "not",
]
COMPLIANCE_STARTERS = [
    "Sure", "Here", "Certainly", "Of", "The", "To",
    "Let", "Below", "This", "def", "import",
]


def find_last_real_token(attention_mask):
    """Index of last non-padding token."""
    m = attention_mask[0] if attention_mask.dim() > 1 else attention_mask
    nz = m.nonzero()
    return int(nz.max().item()) if len(nz) > 0 else 0


def load_datasets(model_key, sample_limit=0):
    """Load refusal + cyber datasets, return flat dicts."""
    base = Path(__file__).resolve().parent.parent / "datasets"
    refusal_all = [json.loads(l) for l in open(
        base / "refusal_probes" / model_key / "attacks_full.jsonl")]
    cyber_train = [json.loads(l) for l in open(
        base / "cyber_probes" / "train.jsonl")]
    cyber_test = [json.loads(l) for l in open(
        base / "cyber_probes" / "test.jsonl")]
    
    if sample_limit > 0:
        refusal_all = refusal_all[:sample_limit]
        # Important: keep proportional train/test for cyber
        train_lim = max(1, int(sample_limit * 0.7))
        test_lim = max(1, sample_limit - train_lim)
        cyber_train = cyber_train[:train_lim]
        cyber_test = cyber_test[:test_lim]

    return {
        "refusal": refusal_all,
        "cyber": cyber_train + cyber_test,
    }


# ---------- task definitions (the 5 scored tasks) ----------
CYBER_PROBE_DEFS = {
    "cyber_probe1": {
        "pos_cat": {"dual_use"},
        "neg_cat": {"benign"},
    },
    "cyber_probe2": {
        "pos_cat": {"high_risk_dual_use"},
        "neg_cat": {"dual_use", "benign"},
    },
    "cyber_probe3": {
        "pos_cat": {"prohibited"},
        "neg_cat": {"high_risk_dual_use", "dual_use", "benign"},
    },
}


def get_task_samples(datasets, task_name, model_key):
    """Return (samples_list, label_fn, prompt_key) for a task."""
    if task_name.startswith("refusal"):
        samples = datasets["refusal"]
        return samples, (lambda s: s["is_refusal"]), "attack_prompt"
    else:
        pdef = CYBER_PROBE_DEFS[task_name]
        samples = [s for s in datasets["cyber"]
                   if s["category"] in pdef["pos_cat"] | pdef["neg_cat"]]
        label_fn = lambda s, _p=pdef: s["category"] in _p["pos_cat"]
        return samples, label_fn, "prompt"


# ---------- forward pass + caching ----------
def forward_and_cache(model, tokenizer, all_samples, prompt_key_map,
                      layer_idxs, use_chunked):
    """Forward all unique samples, cache last-token residuals per layer.

    Args:
        all_samples: list of sample dicts (deduplicated by sample_id)
        prompt_key_map: dict mapping sample_id -> prompt_key
        layer_idxs: list of layer indices to extract
    Returns:
        cache: dict sid -> {last_tok: (n_layers_sel, d_model), n_tokens: int}
    """
    cache = {}
    cm = chunked_sdpa_scope() if use_chunked else None
    if cm is not None:
        cm.__enter__()

    t_start = time.time()
    try:
        for i, s in enumerate(all_samples):
            sid = s["sample_id"]
            if sid in cache:
                continue

            pk = prompt_key_map[sid]
            prompt = s[pk]
            txt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            enc = tokenizer(txt, return_tensors="pt").to(DEVICE)
            ids, attn = enc.input_ids, enc.attention_mask
            torch.cuda.empty_cache()
            try:
                with torch.inference_mode(): # Use inference_mode instead of no_grad for max memory savings
                    out = model(input_ids=ids, attention_mask=attn,
                                output_hidden_states=True, return_dict=True)

                hs = out.hidden_states
                last_idx = find_last_real_token(attn)
                last_tok = torch.stack(
                    [hs[k][0, last_idx, :] for k in layer_idxs], dim=0
                ).cpu().float()  # (n_layers_sel, d_model)

                cache[sid] = {"last_tok": last_tok, "n_tokens": int(ids.shape[1])}
                del out, hs
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                print(f"  [{i+1}/{len(all_samples)}] {sid}: SKIPPING DUE TO OOM (n_tok={int(ids.shape[1])})", flush=True)
            finally:
                del enc, ids, attn
                torch.cuda.empty_cache()

            if (i + 1) % 20 == 0 or (i + 1) == len(all_samples):
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(all_samples) - (i + 1)) / max(rate, 1e-3) / 60
                print(f"  [{i+1}/{len(all_samples)}] {sid}: "
                      f"n_tok={ids.shape[1]} | {rate:.2f}/s eta={eta:.1f}min",
                      flush=True)
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)

    print(f"  Cached {len(cache)} samples in "
          f"{(time.time() - t_start)/60:.1f} min", flush=True)
    return cache


# ---------- direction computation ----------
def compute_direction(cache, samples, label_fn):
    """Compute mean-difference direction per layer from cached residuals."""
    sum_pos, sum_neg = None, None
    n_pos, n_neg = 0, 0

    for s in samples:
        sid = s["sample_id"]
        if sid not in cache:
            continue
        r = cache[sid]["last_tok"]  # (n_layers, d_model)
        if label_fn(s):
            sum_pos = r if sum_pos is None else sum_pos + r
            n_pos += 1
        else:
            sum_neg = r if sum_neg is None else sum_neg + r
            n_neg += 1

    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Empty class: n_pos={n_pos} n_neg={n_neg}")

    direction = (sum_pos / n_pos) - (sum_neg / n_neg)  # (n_layers, d_model)
    norms = direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    direction_normed = direction / norms
    print(f"    n_pos={n_pos} n_neg={n_neg} "
          f"norm_range=[{norms.min():.2f}, {norms.max():.2f}]", flush=True)
    return direction_normed, norms.squeeze()


# ---------- feature extraction ----------
def compute_logit_lens_features(last_tok_residuals, W_U,
                                refusal_ids, compliance_ids):
    """Compute logit-lens features from last-token residuals.

    Args:
        last_tok_residuals: (n_layers, d_model)
        W_U: (vocab_size, d_model)
    Returns:
        refusal_probs: (n_layers,) — sum of softmax probs on refusal tokens
        compliance_probs: (n_layers,) — sum on compliance tokens
    """
    # (n_layers, vocab_size) = (n_layers, d_model) @ (d_model, vocab_size)
    logits = last_tok_residuals @ W_U.T
    # Softmax per layer
    probs = F.softmax(logits, dim=-1)  # (n_layers, vocab_size)
    refusal_probs = probs[:, refusal_ids].sum(dim=-1)      # (n_layers,)
    compliance_probs = probs[:, compliance_ids].sum(dim=-1)  # (n_layers,)
    return refusal_probs.numpy(), compliance_probs.numpy()


def extract_features_for_task(cache, samples, label_fn, direction,
                              W_U, refusal_ids, compliance_ids, n_layers_sel):
    """Build feature matrix for one task.

    Returns: X (n_samples, n_features), y (n_samples,), ids (n_samples,),
             splits (n_samples,), feature_names (n_features,)
    """
    X_rows, y_list, id_list, split_list = [], [], [], []

    for s in samples:
        sid = s["sample_id"]
        if sid not in cache:
            continue

        r = cache[sid]["last_tok"]  # (n_layers, d_model)
        n_tok = cache[sid]["n_tokens"]

        # --- Direction-based features (per layer) ---
        dir_loading = (r * direction).sum(dim=-1).numpy()  # (n_layers,)

        # --- Logit lens features (per layer) ---
        ref_p, comp_p = compute_logit_lens_features(
            r, W_U, refusal_ids, compliance_ids)

        # --- Derived features ---
        dir_diff = np.diff(dir_loading)
        transition_layer = float(np.argmax(np.abs(dir_diff)))
        peak_layer = float(np.argmax(dir_loading))
        total_loading = float(dir_loading.sum())
        max_ratio = float(np.max(
            np.log(ref_p.clip(1e-10) / comp_p.clip(1e-10))))

        # --- Assemble feature vector ---
        feats = np.concatenate([
            dir_loading,                              # n_layers
            ref_p,                                    # n_layers
            comp_p,                                   # n_layers
            [transition_layer, peak_layer, total_loading,
             max_ratio, float(n_tok)],                # 5 derived
        ])
        X_rows.append(feats)
        y_list.append(1 if label_fn(s) else 0)
        id_list.append(sid)
        split_list.append(s.get("split", "unknown"))

    X = np.stack(X_rows, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Feature names
    names = []
    for prefix in ["dir_loading", "logit_refusal", "logit_compliance"]:
        for l in range(n_layers_sel):
            names.append(f"{prefix}_L{l}")
    names += ["transition_layer", "peak_layer", "total_loading",
              "max_logit_ratio", "n_tokens"]

    return X, y, id_list, split_list, names


# ---------- main ----------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_key", choices=list(HF_REPOS.keys()),
                    default="gemma4_31b")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--layers", default="0:62:4",
                    help="Layer-spec. Default every-4th for Gemma.")
    ap.add_argument("--out_dir", default="./hybrid_artifacts")
    ap.add_argument("--sample_limit", type=int, default=0,
                    help="Limit samples per dataset (0 = all, useful for testing)")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) / args.model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Hybrid Pipeline: {args.model_key} layers={args.layers} ===",
          flush=True)

    # --- Load model ---
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = resolve_model_path(args.model_key, args.model_path)
    print(f"Loading model from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=DEVICE, trust_remote_code=True)
    model.eval()

    n_layers = len(model.model.layers) if hasattr(model.model, "layers") else 64
    layer_idxs = parse_layers(args.layers, n_layers)
    n_layers_sel = len(layer_idxs)
    print(f"n_layers={n_layers}, extracting {n_layers_sel}: {layer_idxs}",
          flush=True)

    # --- Save W_U (unembedding matrix) ---
    W_U = model.lm_head.weight.detach().cpu().float()  # (vocab, d_model)
    torch.save(W_U, str(out_dir / "W_U.pt"))
    print(f"Saved W_U: {W_U.shape}", flush=True)

    # --- Resolve token IDs for logit lens ---
    def resolve_token_ids(words):
        ids = []
        for w in words:
            toks = tokenizer.encode(w, add_special_tokens=False)
            if toks:
                ids.append(toks[0])
        return list(set(ids))

    refusal_ids = resolve_token_ids(REFUSAL_STARTERS)
    compliance_ids = resolve_token_ids(COMPLIANCE_STARTERS)
    json.dump({"ids": refusal_ids, "words": REFUSAL_STARTERS},
              open(out_dir / "refusal_token_ids.json", "w"))
    json.dump({"ids": compliance_ids, "words": COMPLIANCE_STARTERS},
              open(out_dir / "compliance_token_ids.json", "w"))
    print(f"Refusal tokens: {len(refusal_ids)}, "
          f"Compliance tokens: {len(compliance_ids)}", flush=True)

    # --- Load datasets ---
    datasets = load_datasets(args.model_key, args.sample_limit)
    print(f"Loaded: refusal={len(datasets['refusal'])}, "
          f"cyber={len(datasets['cyber'])}", flush=True)

    # --- Collect all unique samples + their prompt keys ---
    all_samples = {}
    prompt_key_map = {}
    for s in datasets["refusal"]:
        all_samples[s["sample_id"]] = s
        prompt_key_map[s["sample_id"]] = "attack_prompt"
    for s in datasets["cyber"]:
        all_samples[s["sample_id"]] = s
        prompt_key_map[s["sample_id"]] = "prompt"

    all_samples_list = list(all_samples.values())
    print(f"Total unique samples to forward: {len(all_samples_list)}", flush=True)

    # --- Forward pass: cache all last-token residuals ---
    use_chunked = (args.model_key == "gemma4_31b")
    cache = forward_and_cache(
        model, tokenizer, all_samples_list, prompt_key_map,
        layer_idxs, use_chunked)

    # Free model VRAM
    del model
    torch.cuda.empty_cache()
    print("Model freed from GPU", flush=True)

    # --- Per-task: compute directions + extract features ---
    task_names = [f"refusal_{args.model_key}",
                  "cyber_probe1", "cyber_probe2", "cyber_probe3"]

    for task_name in task_names:
        print(f"\n=== Task: {task_name} ===", flush=True)
        samples, label_fn, prompt_key = get_task_samples(
            datasets, task_name, args.model_key)

        # Train samples for direction
        train_samples = [s for s in samples if s.get("split") == "train"]
        print(f"  Total samples: {len(samples)}, train: {len(train_samples)}",
              flush=True)

        # Compute direction from train
        try:
            direction, dir_norms = compute_direction(
                cache, train_samples, label_fn)
            torch.save({"direction": direction, "norms": dir_norms,
                         "layer_idxs": layer_idxs, "task": task_name},
                        str(out_dir / f"direction_{task_name}.pt"))
        except ValueError as e:
            print(f"  Skipping task {task_name}: {e}", flush=True)
            continue

        # Per-layer AUC with direction only (diagnostic)
        print("  Per-layer direction AUC (train):", flush=True)
        _print_per_layer_auc(cache, train_samples, label_fn, direction,
                             layer_idxs)

        # Extract features
        X, y, ids, splits, feat_names = extract_features_for_task(
            cache, samples, label_fn, direction,
            W_U, refusal_ids, compliance_ids, n_layers_sel)

        np.savez(str(out_dir / f"features_{task_name}.npz"),
                 X=X, y=y, sample_ids=ids, splits=splits,
                 feature_names=feat_names, layer_idxs=layer_idxs)
        print(f"  Saved features: X={X.shape} "
              f"(pos={y.sum()}, neg={len(y)-y.sum()})", flush=True)

    # Save layer config
    json.dump({"layer_idxs": layer_idxs, "n_layers_model": n_layers,
               "model_key": args.model_key, "layers_spec": args.layers},
              open(out_dir / "config.json", "w"), indent=2)

    print(f"\n=== DONE. Artifacts in {out_dir} ===", flush=True)


def _print_per_layer_auc(cache, samples, label_fn, direction, layer_idxs):
    """Quick diagnostic: AUC of dot(residual, direction) per layer."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("    (sklearn not available, skipping)", flush=True)
        return

    n_layers = direction.shape[0]
    scores_per_layer = [[] for _ in range(n_layers)]
    labels = []
    for s in samples:
        if s["sample_id"] not in cache:
            continue
        r = cache[s["sample_id"]]["last_tok"]
        labels.append(1 if label_fn(s) else 0)
        for l in range(n_layers):
            scores_per_layer[l].append(
                float((r[l] * direction[l]).sum()))

    labels = np.array(labels)
    if len(set(labels)) < 2:
        print("    (single class, skipping)", flush=True)
        return

    for l in range(n_layers):
        auc = roc_auc_score(labels, scores_per_layer[l])
        bar = "█" * int(auc * 40)
        print(f"    L{layer_idxs[l]:3d}: AUC={auc:.3f} {bar}", flush=True)


if __name__ == "__main__":
    main()
