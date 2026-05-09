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
        --out_dir /scratch/hybrid \
        --num_gpus 4 --batch_size 8
"""
import os, sys, json, time, argparse
import math
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from chunked_sdpa import chunked_sdpa_scope
from extract_residuals import resolve_model_path, parse_layers, HF_REPOS

# ---------- token sets for logit lens ----------
REFUSAL_STARTERS = [
    "I", "Sorry", "cannot", "unable", "Unfortunately", "apologize",
    "can't", "won't", "No", "not",
]
COMPLIANCE_STARTERS = [
    "Sure", "Here", "Certainly", "Of", "The", "To",
    "Let", "Below", "This", "def", "import",
]


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
                      layer_idxs, use_chunked, batch_size=8, device="cuda:0"):
    """Forward unique samples with batching, cache last-token residuals."""
    cache = {}
    cm = chunked_sdpa_scope() if use_chunked else None
    if cm is not None:
        cm.__enter__()

    # For reliable last-token extraction across batched generations
    tokenizer.padding_side = "right"

    t_start = time.time()
    try:
        # We handle batched processing, with a fallback to size 1 if OOM occurs
        i = 0
        while i < len(all_samples):
            # Form batch
            batch_samples = []
            while len(batch_samples) < batch_size and i < len(all_samples):
                if all_samples[i]["sample_id"] not in cache:
                    batch_samples.append(all_samples[i])
                i += 1
                
            if not batch_samples:
                continue
                
            prompts = [s[prompt_key_map[s["sample_id"]]] for s in batch_samples]
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True,
                ) for p in prompts
            ]
            
            enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
            ids, attn = enc.input_ids, enc.attention_mask
            
            torch.cuda.empty_cache()
            
            try:
                with torch.inference_mode():
                    out = model(input_ids=ids, attention_mask=attn,
                                output_hidden_states=True, return_dict=True)

                hs = out.hidden_states
                
                # Length of each sequence ignoring right-padding
                lengths = attn.sum(dim=1) - 1
                
                for b_idx, s in enumerate(batch_samples):
                    sid = s["sample_id"]
                    n_tok = int(lengths[b_idx].item() + 1)
                    last_idx = int(lengths[b_idx].item())
                    
                    last_tok = torch.stack(
                        [hs[k][b_idx, last_idx, :] for k in layer_idxs], dim=0
                    ).cpu().float()  # (n_layers_sel, d_model)

                    cache[sid] = {"last_tok": last_tok, "n_tokens": n_tok}

                del out, hs

            except (torch.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower() or batch_size == 1:
                    raise
                print(f"  [{device}] OOM with batch_size={batch_size}, retrying sequentially...", flush=True)
                # Rewind and retry with batch size 1
                i -= len(batch_samples)
                # temporarily drop batch size for this failing set
                for fallback_s in batch_samples:
                    if fallback_s["sample_id"] in cache: continue
                    try:
                        f_txt = tokenizer.apply_chat_template(
                            [{"role": "user", "content": fallback_s[prompt_key_map[fallback_s["sample_id"]]]}],
                            tokenize=False, add_generation_prompt=True,
                        )
                        f_enc = tokenizer(f_txt, return_tensors="pt").to(device)
                        with torch.inference_mode():
                            f_out = model(input_ids=f_enc.input_ids, attention_mask=f_enc.attention_mask,
                                          output_hidden_states=True, return_dict=True)
                        last_idx = int(f_enc.attention_mask.sum().item() - 1)
                        f_last_tok = torch.stack(
                            [f_out.hidden_states[k][0, last_idx, :] for k in layer_idxs], dim=0
                        ).cpu().float()
                        cache[fallback_s["sample_id"]] = {"last_tok": f_last_tok, "n_tokens": last_idx + 1}
                        del f_out
                    except Exception as fallback_e:
                        print(f"  [{device}] SKIPPING {fallback_s['sample_id']} due to OOM: {fallback_e}", flush=True)
                    finally:
                        del f_enc
                        torch.cuda.empty_cache()
                # Skip the batch we just manually processed
                i += len(batch_samples)
            finally:
                del enc, ids, attn
                torch.cuda.empty_cache()

            if i % (batch_size * 5) < batch_size or i == len(all_samples):
                elapsed = time.time() - t_start
                rate = i / elapsed
                eta = (len(all_samples) - i) / max(rate, 1e-3) / 60
                print(f"  [{device}] [{i}/{len(all_samples)}] "
                      f"rate={rate:.2f}/s eta={eta:.1f}min", flush=True)
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)

    print(f"  [{device}] Cached {len(cache)} samples in "
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
    """Compute logit-lens features from last-token residuals."""
    logits = last_tok_residuals @ W_U.T
    probs = F.softmax(logits, dim=-1)  # (n_layers, vocab_size)
    refusal_probs = probs[:, refusal_ids].sum(dim=-1)      # (n_layers,)
    compliance_probs = probs[:, compliance_ids].sum(dim=-1)  # (n_layers,)
    return refusal_probs.numpy(), compliance_probs.numpy()


def extract_features_for_task(cache, samples, label_fn, direction,
                              W_U, refusal_ids, compliance_ids, n_layers_sel):
    """Build feature matrix for one task."""
    X_rows, y_list, id_list, split_list = [], [], [], []

    for s in samples:
        sid = s["sample_id"]
        if sid not in cache:
            continue

        r = cache[sid]["last_tok"]  # (n_layers, d_model)
        n_tok = cache[sid]["n_tokens"]

        dir_loading = (r * direction).sum(dim=-1).numpy()  # (n_layers,)
        ref_p, comp_p = compute_logit_lens_features(
            r, W_U, refusal_ids, compliance_ids)

        dir_diff = np.diff(dir_loading)
        transition_layer = float(np.argmax(np.abs(dir_diff)))
        peak_layer = float(np.argmax(dir_loading))
        total_loading = float(dir_loading.sum())
        max_ratio = float(np.max(
            np.log(ref_p.clip(1e-10) / comp_p.clip(1e-10))))

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

    names = []
    for prefix in ["dir_loading", "logit_refusal", "logit_compliance"]:
        for l in range(n_layers_sel):
            names.append(f"{prefix}_L{l}")
    names += ["transition_layer", "peak_layer", "total_loading",
              "max_logit_ratio", "n_tokens"]

    return X, y, id_list, split_list, names


# ---------- multiprocessing worker ----------
def extract_worker(rank, args, all_samples_chunk, prompt_key_map, layer_idxs, use_chunked, out_dir):
    """Worker process that loads model on cuda:<rank> and extracts features."""
    device = f"cuda:{rank}"
    print(f"[{device}] Worker started, processing {len(all_samples_chunk)} samples...", flush=True)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = resolve_model_path(args.model_key, args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device, trust_remote_code=True)
    model.eval()

    cache = forward_and_cache(
        model, tokenizer, all_samples_chunk, prompt_key_map,
        layer_idxs, use_chunked, args.batch_size, device)
        
    tmp_path = out_dir / f"tmp_cache_shard_{rank}.pt"
    torch.save(cache, tmp_path)
    print(f"[{device}] Saved {len(cache)} items to {tmp_path}", flush=True)


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
    ap.add_argument("--num_gpus", type=int, default=1,
                    help="Number of GPUs to run data-parallel extraction on")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Batch size per GPU for extraction")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) / args.model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Hybrid Pipeline: {args.model_key} layers={args.layers} ===", flush=True)
    print(f"GPUs: {args.num_gpus}, Batch Size: {args.batch_size}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = resolve_model_path(args.model_key, args.model_path)
    
    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Extract W_U quickly on cuda:0
    print(f"Loading model briefly on cuda:0 to extract W_U...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0", trust_remote_code=True)
    model.eval()

    n_layers = len(model.model.layers) if hasattr(model.model, "layers") else 64
    layer_idxs = parse_layers(args.layers, n_layers)
    n_layers_sel = len(layer_idxs)

    W_U = model.lm_head.weight.detach().cpu().float()  # (vocab, d_model)
    torch.save(W_U, str(out_dir / "W_U.pt"))
    print(f"Saved W_U: {W_U.shape}", flush=True)

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

    # Free model VRAM
    del model
    torch.cuda.empty_cache()
    print("Model freed from GPU\n", flush=True)

    # 3. Load datasets and prepare samples
    datasets = load_datasets(args.model_key, args.sample_limit)
    print(f"Loaded: refusal={len(datasets['refusal'])}, cyber={len(datasets['cyber'])}", flush=True)

    all_samples = {}
    prompt_key_map = {}
    for s in datasets["refusal"]:
        all_samples[s["sample_id"]] = s
        prompt_key_map[s["sample_id"]] = "attack_prompt"
    for s in datasets["cyber"]:
        all_samples[s["sample_id"]] = s
        prompt_key_map[s["sample_id"]] = "prompt"

    all_samples_list = list(all_samples.values())
    print(f"Total unique samples to forward: {len(all_samples_list)}\n", flush=True)
    use_chunked = (args.model_key == "gemma4_31b")

    # 4. Multiprocessing Extraction
    if args.num_gpus > 1:
        mp.set_start_method("spawn", force=True)
        chunk_size = math.ceil(len(all_samples_list) / args.num_gpus)
        processes = []
        
        for rank in range(args.num_gpus):
            chunk = all_samples_list[rank * chunk_size : (rank + 1) * chunk_size]
            if not chunk: continue
            
            p = mp.Process(target=extract_worker, args=(
                rank, args, chunk, prompt_key_map, layer_idxs, use_chunked, out_dir
            ))
            p.start()
            processes.append((rank, p))
            
        for rank, p in processes:
            p.join()
            if p.exitcode != 0:
                print(f"ERROR: Worker {rank} failed with exitcode {p.exitcode}", flush=True)
                sys.exit(1)
                
        # Merge caches
        cache = {}
        for rank in range(args.num_gpus):
            tmp_path = out_dir / f"tmp_cache_shard_{rank}.pt"
            if tmp_path.exists():
                shard_cache = torch.load(tmp_path, weights_only=False)
                cache.update(shard_cache)
                tmp_path.unlink() # Clean up
        print(f"\nMerged {len(cache)} cached samples from all workers.", flush=True)
    else:
        # Sequential on 1 GPU
        print("Running sequential extraction on cuda:0...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda:0", trust_remote_code=True)
        model.eval()
        cache = forward_and_cache(
            model, tokenizer, all_samples_list, prompt_key_map,
            layer_idxs, use_chunked, args.batch_size, "cuda:0")
        del model
        torch.cuda.empty_cache()

    # 5. Compute Directions & Features
    task_names = [f"refusal_{args.model_key}", "cyber_probe1", "cyber_probe2", "cyber_probe3"]

    for task_name in task_names:
        print(f"\n=== Task: {task_name} ===", flush=True)
        samples, label_fn, prompt_key = get_task_samples(
            datasets, task_name, args.model_key)

        train_samples = [s for s in samples if s.get("split") == "train"]
        
        try:
            direction, dir_norms = compute_direction(
                cache, train_samples, label_fn)
            torch.save({"direction": direction, "norms": dir_norms,
                         "layer_idxs": layer_idxs, "task": task_name},
                        str(out_dir / f"direction_{task_name}.pt"))
        except ValueError as e:
            print(f"  Skipping task {task_name}: {e}", flush=True)
            continue

        print("  Per-layer direction AUC (train):", flush=True)
        _print_per_layer_auc(cache, train_samples, label_fn, direction, layer_idxs)

        X, y, ids, splits, feat_names = extract_features_for_task(
            cache, samples, label_fn, direction,
            W_U, refusal_ids, compliance_ids, n_layers_sel)

        np.savez(str(out_dir / f"features_{task_name}.npz"),
                 X=X, y=y, sample_ids=ids, splits=splits,
                 feature_names=feat_names, layer_idxs=layer_idxs)
        print(f"  Saved features: X={X.shape} (pos={y.sum()}, neg={len(y)-y.sum()})", flush=True)

    json.dump({"layer_idxs": layer_idxs, "n_layers_model": n_layers,
               "model_key": args.model_key, "layers_spec": args.layers},
              open(out_dir / "config.json", "w"), indent=2)

    print(f"\n=== DONE. Artifacts in {out_dir} ===", flush=True)


def _print_per_layer_auc(cache, samples, label_fn, direction, layer_idxs):
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
        return

    for l in range(n_layers):
        auc = roc_auc_score(labels, scores_per_layer[l])
        bar = "█" * int(auc * 40)
        print(f"    L{layer_idxs[l]:3d}: AUC={auc:.3f} {bar}", flush=True)


if __name__ == "__main__":
    main()
