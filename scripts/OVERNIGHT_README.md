# Overnight chains — fire and forget

Three master scripts, one per pod. Each waits for its pod's currently-running
job to finish, then queues additional experiments sequentially. **Failure of
one experiment does not stop the chain.**

## What each script does

### `overnight_dev.sh` — PRE K-sweep (~5h)
1. Wait for current PRE K=3 to finish
2. Run PRE K=7 on n=81 (~2.5h)
3. Run PRE K=10 on n=81 (~3h)
- **Goal**: build the diminishing-returns curve `flip_rate(K)` for the slide

### `overnight_cyber.sh` — fine-grained layer sweep (~4h)
1. Wait for current 5-layer sweep (16/32/44/48/56) to finish
2. Build single-layer direction files for 9 more layers
3. Run patch_agent at layers 4, 8, 12, 20, 24, 28, 36, 40, 52 (each n=20)
- **Goal**: full layer-sweep curve at every-4-layer granularity. The money figure.

### `overnight_qwen.sh` — whitened + PRE-whitened + cyber cascade + multilayer (~6h)
1. Wait for whitened direction file to be saved
2. Whitened ablation smoke (n=2)
3. Whitened ablation full (n=20)
4. PRE n=81 using whitened direction as scorer
5. Cyber-cascade flip stretch goal (PRE on n=30 cyber-prohibited prompts using cyber-3 probe direction)
6. Multilayer mean-pool ablation (n=20)
- **Goal**: test multiple direction-construction methods as scorers; attempt the hackathon stretch goal

## Launch sequence

Each script needs to be launched on its respective pod. The first script (whatever's running NOW) is already going on each pod — these master scripts wait for it then queue up additional work.

### dev-pod (after PRE n=81 K=3 is running)

```bash
cd /scratch/mechhack
nohup bash scripts/overnight_dev.sh > logs/overnight_dev.log 2>&1 &
echo "DEV-MASTER PID=$!"
date
```

### cyber-pod (after layer sweep is running)

```bash
cd /scratch/mechhack
nohup bash scripts/overnight_cyber.sh > logs/overnight_cyber.log 2>&1 &
echo "CYBER-MASTER PID=$!"
date
```

### qwen-pod (after whitened build is running)

```bash
cd /scratch/mechhack
nohup bash scripts/overnight_qwen.sh > logs/overnight_qwen.log 2>&1 &
echo "QWEN-MASTER PID=$!"
date
```

## Verification before sleep

```bash
# On any pod
for f in dev cyber qwen; do
    echo "=== overnight_$f ==="
    ls -la /scratch/mechhack/logs/overnight_$f.log 2>/dev/null
done
```

Should see three log files growing. Then on each pod:

```bash
ps aux | grep -E "overnight_|patch_agent|run_pre_resumable" | grep -v grep
```

Should see at least 1 master process AND 1 worker process (the currently-running experiment) per pod.

## Tomorrow morning

```bash
# All experiment outputs in one place
ls -la /scratch/mechhack/edit_eval/

# Layer sweep summary
python -c "
import json, glob
files = sorted(glob.glob('/scratch/mechhack/edit_eval/patch_layer_L*.json'))
print(f'{\"layer\":>6} | {\"baseline\":>8} | {\"ablated\":>7} | {\"random\":>6}')
print('-' * 40)
def L_of(f): return int(f.split('_L')[-1].split('.')[0])
for f in sorted(files, key=L_of):
    try:
        d = json.load(open(f))
        agg = d['aggregate']
        L = L_of(f)
        print(f'L{L:>4} | {agg[\"baseline_compliance\"]:>8} | {agg[\"ablated_compliance\"]:>7} | {agg[\"random_compliance\"]:>6}')
    except Exception as e:
        print(f'{f}: error {e}')
"

# K-sweep summary
for K in 3 7 10; do
    F=edit_eval/pre_${K}.json
    [ "$K" = "3" ] && F=edit_eval/pre_n81.json
    [ "$K" = "7" ] && F=edit_eval/pre_k7_n81.json
    [ "$K" = "10" ] && F=edit_eval/pre_k10_n81.json
    echo "=== K=$K ==="
    python -c "
import json
try:
    d = json.load(open('/scratch/mechhack/$F'))
    a = d['aggregate']
    print(f\"  Pr(model|edit) = {a['Pr_model_given_edit']:.3f} CI={a['Pr_model_given_edit_CI95']}\")
    print(f\"  Pr(f|edit)     = {a['Pr_f_given_edit']:.3f}\")
    print(f\"  intent_pres    = {a['intent_preservation_rate']:.3f}\")
except Exception as e:
    print(f'  (error: {e})')
"
done

# Whitened, multilayer, cyber-cascade summaries
for f in patch_whitened_n20 patch_multilayer_meanpool_n20 pre_whitened_n81 pre_cyber3_cascade; do
    echo "=== $f ==="
    python -c "
import json
try:
    d = json.load(open('/scratch/mechhack/edit_eval/${f}.json'))
    print(json.dumps(d.get('aggregate', {}), indent=2))
except Exception as e:
    print(f'(error: {e})')
"
done
```

Paste those summaries to chat. Then audit:

```bash
# Audit n=81 PRE compliances in batches of 25
cd /scratch/mechhack
python scripts/audit_helper.py --input edit_eval/pre_n81.json --start 0
python scripts/audit_helper.py --input edit_eval/pre_n81.json --start 25
python scripts/audit_helper.py --input edit_eval/pre_n81.json --start 50

# K=7 and K=10 audits
python scripts/audit_helper.py --input edit_eval/pre_k7_n81.json --start 0
# etc
```

## If a chain crashed overnight

Each `overnight_*.sh` script logs its progress to `logs/overnight_*.log`. Check
the last line of each to see what step it died on:

```bash
tail -5 /scratch/mechhack/logs/overnight_dev.log
tail -5 /scratch/mechhack/logs/overnight_cyber.log
tail -5 /scratch/mechhack/logs/overnight_qwen.log
```

The individual experiment outputs are checkpointed per-prompt, so partial results are preserved.
