"""XGBoost probes on saved activations. Reuses the same .pt files used by train_probe.py."""
import json, glob, torch, numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import xgboost as xgb

EXTRACTS = {
    "refusal_gemma4_31b": "/scratch/mechhack/extracts/gemma4_31b",
    "refusal_qwen36":     "/scratch/mechhack/extracts/qwen36",
    "cyber_du":           "/scratch/mechhack/extracts/cyber_gemma4_31b",
    "cyber_hdu":          "/scratch/mechhack/extracts/cyber_gemma4_31b",
    "cyber_prohib":       "/scratch/mechhack/extracts/cyber_gemma4_31b",
}

manifest = json.load(open("/scratch/mechhack/extracts/unified_manifest.json"))

def load_features(extracts_dir, samples, label_fn):
    """Load last-token activation per sample as feature vector."""
    X, y, splits = [], [], []
    for s in samples:
        path = Path(extracts_dir) / f"{s['sample_id']}.pt"
        if not path.exists(): continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        res = d["residuals"]              # (n_layers, n_tokens, d_model)
        mask = d["attention_mask"].bool()
        last = mask.sum().item() - 1
        feat = res[0, last, :].float().numpy()
        X.append(feat)
        y.append(label_fn(s))
        splits.append(s["split"])
    return np.array(X), np.array(y), np.array(splits)

def task_specs():
    yield ("refusal_gemma4_31b", manifest["refusal_samples"]["gemma4_31b"],
           lambda s: 1 if s["is_refusal"] else 0, EXTRACTS["refusal_gemma4_31b"])
    yield ("refusal_qwen36", manifest["refusal_samples"]["qwen36"],
           lambda s: 1 if s["is_refusal"] else 0, EXTRACTS["refusal_qwen36"])
    cyber = manifest["cyber_samples"]
    for cls, key in [("dual_use","cyber_du"), ("high_risk_dual_use","cyber_hdu"),
                     ("prohibited","cyber_prohib")]:
        # Match train_probe.py's balancing — pos vs balanced negatives
        pos = [s for s in cyber if s["label"] == cls]
        neg_all = [s for s in cyber if s["label"] != cls]
        other_cats = sorted({s["label"] for s in neg_all})
        n_per = len(pos) // len(other_cats)
        import random
        rng = random.Random(42 + hash(cls) % 1000)
        neg = []
        for c in other_cats:
            pool = [s for s in neg_all if s["label"] == c]
            neg += rng.sample(pool, min(n_per, len(pool)))
        samples = pos + neg
        yield (f"cyber_{cls}", samples,
               (lambda s, _cls=cls: 1 if s["label"] == _cls else 0),
               EXTRACTS[key])

results = {}
print(f"{'Task':<30} {'XGBoost AUC':<12}")
print("=" * 50)
for task_name, samples, label_fn, extracts_dir in task_specs():
    X, y, splits = load_features(extracts_dir, samples, label_fn)
    if len(X) == 0:
        print(f"{task_name:<30} (no data)"); continue
    train_mask = splits == "train"
    test_mask = splits == "test"
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        objective="binary:logistic", eval_metric="auc",
        tree_method="hist", n_jobs=8, random_state=42,
    )
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    results[task_name] = auc
    print(f"{task_name:<30} {auc:.4f}     (n_train={len(X_train)}, n_test={len(X_test)})")

print("\nMean AUC across hackathon 5:")
keys = ["refusal_gemma4_31b","refusal_qwen36","cyber_dual_use","cyber_high_risk_dual_use","cyber_prohibited"]
m = sum(results.get(k,0) for k in keys) / 5
print(f"  {m:.4f}")

# Save
import json
json.dump(results, open("/scratch/mechhack/probes/results/xgboost_results.json", "w"), indent=2)
