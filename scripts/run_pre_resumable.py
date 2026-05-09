"""Resumable PRE driver. Wraps pre_agent.run_pre_on_prompt with
per-prompt checkpointing AND restart-from-existing-output.

If killed, rerun the same command. The wrapper checks the output JSON for
already-completed sample_ids and skips them.

Usage:
    export AIAAS_KEY=sk-...
    python scripts/run_pre_resumable.py \\
        --eval_set datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl \\
        --direction /scratch/hybrid/gemma4_31b/direction_refusal_gemma4_31b.pt \\
        --output edit_eval/pre_n81.json \\
        --limit 81 --k 3
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, "/scratch/mechhack")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--model_path", default="/data/Gemma-4-31B-it")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=81)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--intent_threshold", type=int, default=7)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load existing results if any
    done_ids = set()
    existing = []
    if out_path.exists():
        try:
            with open(out_path) as f:
                prev = json.load(f)
            existing = prev.get("results", [])
            done_ids = {r["sample_id"] for r in existing if "sample_id" in r}
            print(f"[resume] {len(done_ids)} sample(s) already done in {out_path}", flush=True)
        except Exception as e:
            print(f"[resume] could not parse {out_path}: {e}; starting fresh", flush=True)

    # Load eval set
    with open(args.eval_set) as f:
        samples = [json.loads(l) for l in f if l.strip()]
    samples = samples[: args.limit]
    todo = [s for s in samples
            if (s.get("sample_id") or s.get("id")) not in done_ids]
    print(f"[main] {len(samples)} total, {len(done_ids)} done, {len(todo)} todo",
          flush=True)
    if not todo:
        print("[main] nothing to do, exiting", flush=True)
        return

    # GPU pre-flight
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_b, total_b = torch.cuda.mem_get_info(0)
        print(f"[main] GPU0: {free_b/1e9:.1f}/{total_b/1e9:.1f} GB free",
              flush=True)

    # Heavy imports from pre_agent
    from pre_agent import (
        ProbeScorer,
        run_pre_on_prompt,
        aggregate,
        PromptResult,
    )

    scorer = ProbeScorer(
        model_path=args.model_path,
        direction_path=args.direction,
    )

    # Reconstruct PromptResult objects from existing dicts so aggregate()
    # has a uniform list to work over.
    results = []
    for d in existing:
        try:
            kwargs = {
                k: v for k, v in d.items()
                if k in PromptResult.__dataclass_fields__
            }
            results.append(PromptResult(**kwargs))
        except Exception:
            pass

    for i, s in enumerate(todo, 1):
        t0 = time.time()
        try:
            r = run_pre_on_prompt(
                s, scorer,
                k=args.k,
                intent_threshold=args.intent_threshold,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{i}/{len(todo)}] {s.get('sample_id')} CRASHED: {e}",
                  flush=True)
            continue
        results.append(r)
        delta_str = (f"{r.delta_probe:+.3f}"
                     if r.delta_probe is not None else "  N/A")
        print(
            f"[{i}/{len(todo)}] {r.sample_id} dprobe={delta_str} "
            f"intent={r.intent_score or '-'} "
            f"behavior={r.edited_behavior or '-'} "
            f"({time.time()-t0:.1f}s)",
            flush=True,
        )
        # Checkpoint after every prompt
        with open(out_path, "w") as f:
            json.dump(
                {
                    "config": vars(args),
                    "results": [asdict(r) for r in results],
                    "aggregate": aggregate(results),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    print("\n=== FINAL AGGREGATE ===", flush=True)
    print(json.dumps(aggregate(results), indent=2), flush=True)


if __name__ == "__main__":
    main()
