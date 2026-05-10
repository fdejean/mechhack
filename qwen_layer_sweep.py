"""Train probes at each Qwen layer (16, 32, 48), find best AUC."""
import json, torch, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

EXTRACTS = Path("/scratch/mechhack/extracts/qwen36_multilayer/qwen36")
manifest = json.load(open("/scratch/mechhack/extracts/unified_manifest.json"))
samples = manifest["refusal_samples"]["qwen36"]

# Load metadata to find which layer indices were saved
meta_path = EXTRACTS.parent / "extraction_metadata.json"
if not meta_path.exists():
    meta_path = EXTRACTS / "extraction_metadata.json"
meta = json.load(open(meta_path))
layer_idxs = meta["layer_idxs"]
print(f"Extracted layers: {layer_idxs}")

# For each layer, build X/y and train probes
for li, layer_idx in enumerate(layer_idxs):
    X, y, splits = [], [], []
    for s in samples:
        path = EXTRACTS / f"{s['sample_id']}.pt"
        if not path.exists(): continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        res = d["residuals"]                 # (n_layers, n_tokens, d_model)
        mask = d["attention_mask"].bool()
        last = mask.sum().item() - 1
        feat = res[li, last, :].float().numpy()  # ← pick this layer
        X.append(feat)
        y.append(1 if s["is_refusal"] else 0)
        splits.append(s["split"])

    X, y, splits = np.array(X), np.array(y), np.array(splits)
    X_tr, y_tr = X[splits=="train"], y[splits=="train"]
    X_te, y_te = X[splits=="test"], y[splits=="test"]

    # Linear
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(X_tr, y_tr)
    auc_lin = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])

    # XGBoost
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             objective="binary:logistic", tree_method="hist",
                             n_jobs=8, random_state=42)
    clf.fit(X_tr, y_tr)
    auc_xgb = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])

    print(f"Layer {layer_idx:>3}:  Linear AUC = {auc_lin:.4f}   XGBoost AUC = {auc_xgb:.4f}   (n_train={len(X_tr)}, n_test={len(X_te)})")
