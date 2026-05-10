import json
from pathlib import Path

# Load extracts metadata
gemma_meta = json.load(open("/scratch/mechhack/extracts/gemma4_31b/extraction_metadata.json"))
qwen_meta = json.load(open("/scratch/mechhack/extracts/qwen36/extraction_metadata.json"))

extracted_gemma_ids = {s["sample_id"] for s in gemma_meta["samples"]}
extracted_qwen_ids = {s["sample_id"] for s in qwen_meta["samples"]}

# Load refusal labels
def load_jsonl(path):
    return [json.loads(l) for l in open(path)]

gemma_refusal = load_jsonl("/scratch/mechhack/datasets/refusal_probes/gemma4_31b/attacks_full.jsonl")
qwen_refusal = load_jsonl("/scratch/mechhack/datasets/refusal_probes/qwen36/attacks_full.jsonl")

refusal_samples = {
    "gemma4_31b": [s for s in gemma_refusal if s["sample_id"] in extracted_gemma_ids],
    "qwen36": [s for s in qwen_refusal if s["sample_id"] in extracted_qwen_ids],
}

# Cyber: load both train+test, but cyber-Gemma extraction barely started, so this'll be tiny for now
cyber_train = load_jsonl("/scratch/mechhack/datasets/cyber_probes/train.jsonl")
cyber_test = load_jsonl("/scratch/mechhack/datasets/cyber_probes/test.jsonl")
cyber_all = cyber_train + cyber_test

cyber_extracted_ids = set()
cyber_dir = Path("/scratch/mechhack/extracts/cyber_gemma4_31b")
if cyber_dir.exists():
    cyber_extracted_ids = {p.stem for p in cyber_dir.glob("*.pt")}

cyber_samples = [{**s, "label": s["category"]} for s in cyber_all if s["sample_id"] in cyber_extracted_ids]

manifest = {
    "refusal_samples": refusal_samples,
    "cyber_samples": cyber_samples,
}

print(f"Refusal-Gemma: {len(refusal_samples['gemma4_31b'])}")
print(f"Refusal-Qwen:  {len(refusal_samples['qwen36'])}")
print(f"Cyber:         {len(cyber_samples)}")

# Write to BOTH extract dirs (train_probe.py reads one manifest, but expects all data via extracts_dir)
out = "/scratch/mechhack/extracts/unified_manifest.json"
json.dump(manifest, open(out, "w"))
print(f"\nWrote {out}")
