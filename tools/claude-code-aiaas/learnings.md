# Mechhack Learnings

## Setup (done night before hackathon)
- Connected to EPFL cluster via `runai login`
- Project: hackathon-mechhack-kavun, group g06
- Two pods: dev-pod (gpu018, now gpu019) + cyber-pod (gpu005, now gpu019), both on same node
- EPFL VPN required for runai commands from off-campus

## Extraction
- Refusal-Gemma: /scratch/mechhack/extracts/gemma4_31b/ — 840 files
- Refusal-Qwen: /scratch/mechhack/extracts/qwen36/ — 862 files
- Cyber-Gemma: /scratch/mechhack/extracts/cyber_gemma4_31b/gemma4_31b/ — 4627+ files, still running
- Cyber dataset uses `prompt` field, scripts expect `attack_prompt` → fixed via all_fixed.jsonl
- train_probe.py expects unified manifest with refusal_samples + cyber_samples → build_manifest.py handles this
- train_probe.py overwrites metrics.jsonl each run → save to separate files (gemma_metrics.jsonl, qwen_metrics.jsonl) before running next task

## Probes
- Gemma refusal best: AUC 0.914 (attention, batch)
- train_probe.py: linear/MLP/attention × 5 seeds, batch/incremental regimes
- Weights: /scratch/mechhack/probes/weights/

## Git
- Remote: git@github.com:fdejean/mechhack.git
- SSH key generated on cluster, added to GitHub
- Commit as Veronika only (no Claude Code co-author)

## Claude Code (aiaas-claude.sh)
- MiniMax-M2.7 is the only option on AIaaS (no Opus/Anthropic)
- Cold start ~3 min, then ~8-10s/call
- IS_SANDBOX=1 set, Bypass Permissions mode expected
- AIAAS_KEY needed from portal.rcp.epfl.ch/aiaas/keys

## Pods
- Auto-suspend after 2hr idle → runai resume JOB-NAME
- /scratch is persistent PVC, /data is read-only model weights
- 3 GPUs deserved, 2 currently used