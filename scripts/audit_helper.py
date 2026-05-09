"""Print judge-flagged compliances in batches for manual audit.

Usage:
    # PRE results (default mode):
    python scripts/audit_helper.py --input edit_eval/pre_n81.json
    python scripts/audit_helper.py --input edit_eval/pre_n81.json --start 25
    python scripts/audit_helper.py --input edit_eval/pre_n81.json --start 50

    # patch_agent results:
    python scripts/audit_helper.py --input edit_eval/patch_layer_L48.json --mode patch
    python scripts/audit_helper.py --input edit_eval/patch_whitened.json --mode patch

The output is plain text designed to be copy-pasted into a chat for review.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument(
        "--batch", type=int, default=25,
        help="How many flagged samples to print per batch.",
    )
    ap.add_argument(
        "--start", type=int, default=0,
        help="Index of the first flagged sample to show.",
    )
    ap.add_argument(
        "--mode", choices=["pre", "patch"], default="pre",
        help="Which kind of result file this is.",
    )
    ap.add_argument(
        "--max_chars", type=int, default=2000,
        help="Truncate response text to this many characters.",
    )
    args = ap.parse_args()

    data = json.load(open(args.input))
    results = data.get("results", data) if isinstance(data, dict) else data

    if args.mode == "pre":
        flips = [r for r in results
                 if r.get("edited_behavior") == "compliance"]
    else:
        flips = []
        for r in results:
            for cond in ["baseline", "ablated", "random"]:
                if r.get(f"{cond}_label") == "compliance":
                    flips.append({"_cond": cond, **r})

    total = len(flips)
    end = min(args.start + args.batch, total)
    print(f"# Audit batch {args.start}-{end} of {total}\n", flush=True)

    for r in flips[args.start:end]:
        sid = r.get("sample_id", "?")
        if args.mode == "pre":
            dp = r.get("delta_probe")
            dp_str = f"{dp:+.3f}" if isinstance(dp, (int, float)) else "?"
            intent = r.get("intent_score", "?")
            print("=" * 78)
            print(f"=== {sid} | dprobe={dp_str} | intent={intent} ===")
            print("=" * 78)
            print("\n--- ORIGINAL PROMPT (first 400 chars) ---")
            print((r.get("original_prompt") or "")[:400])
            print("\n--- CHOSEN REWRITE (first 600 chars) ---")
            print((r.get("chosen_rewrite") or "")[:600])
            print(f"\n--- EDITED RESPONSE (first {args.max_chars}) ---")
            print((r.get("edited_response") or "")[:args.max_chars])
        else:
            cond = r["_cond"]
            print("=" * 78)
            print(f"=== {sid} | condition={cond} ===")
            print("=" * 78)
            print("\n--- PROMPT (first 400 chars) ---")
            print((r.get("prompt") or r.get("original_prompt") or "")[:400])
            print(f"\n--- {cond.upper()} RESPONSE (first {args.max_chars}) ---")
            print((r.get(f"{cond}_response") or "")[:args.max_chars])
        print()

    print(f"# next: --start {end}", flush=True)


if __name__ == "__main__":
    main()
