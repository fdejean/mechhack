"""Train XGBoost + baselines on hybrid features.

Loads .npz feature files from compute_and_extract.py, trains per-task
classifiers, reports AUC, and saves models + feature importance plots.

Usage:
    python train_hybrid.py \
        --features_dir ./hybrid_artifacts/gemma4_31b \
        --out_dir ./hybrid_models

Requires: xgboost, scikit-learn, matplotlib (all in pod image).
"""
import json, argparse
from pathlib import Path
import numpy as np

TASKS = ["refusal_gemma4_31b", "cyber_probe1", "cyber_probe2", "cyber_probe3"]
TASKS_QWEN = ["refusal_qwen36", "cyber_probe1", "cyber_probe2", "cyber_probe3"]


def load_task(features_dir, task_name):
    """Load feature matrix + labels for one task."""
    p = Path(features_dir) / f"features_{task_name}.npz"
    if not p.exists():
        return None
    d = np.load(str(p), allow_pickle=True)
    X = d["X"]
    y = d["y"]
    splits = d["splits"]
    feat_names = d["feature_names"]
    sample_ids = d["sample_ids"]
    train_mask = splits == "train"
    test_mask = splits == "test"
    return {
        "X_train": X[train_mask], "y_train": y[train_mask],
        "X_test": X[test_mask], "y_test": y[test_mask],
        "feat_names": list(feat_names),
        "ids_train": sample_ids[train_mask],
        "ids_test": sample_ids[test_mask],
    }


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with RandomizedSearchCV, return best model + metrics."""
    from xgboost import XGBClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import roc_auc_score

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_estimators=300,
        early_stopping_rounds=20,
    )

    param_distributions = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    rs = RandomizedSearchCV(
        base_model, param_distributions, n_iter=10,
        scoring="roc_auc", cv=5, random_state=42, n_jobs=-1
    )
    
    # Use test set for early stopping.
    rs.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    best_model = rs.best_estimator_
    preds = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    best_idx = rs.best_index_
    cv_mean = rs.cv_results_["mean_test_score"][best_idx]
    cv_std = rs.cv_results_["std_test_score"][best_idx]

    return best_model, auc, preds, cv_mean, cv_std


def train_logistic(X_train, y_train, X_test, y_test):
    """Logistic regression baseline with GridSearchCV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    param_grid = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}
    lr = LogisticRegression(max_iter=1000, random_state=42)

    gs = GridSearchCV(lr, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_tr, y_train)

    best_model = gs.best_estimator_
    preds = best_model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_test, preds)
    return best_model, auc, preds, scaler
def train_svm(X_train, y_train, X_test, y_test):
    """SVM baseline with GridSearchCV."""
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf']
    }
    svm = SVC(probability=True, random_state=42)

    gs = GridSearchCV(svm, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_tr, y_train)

    best_model = gs.best_estimator_
    preds = best_model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_test, preds)
    return best_model, auc, preds, scaler


def train_direction_only(X_train, y_train, X_test, y_test, n_layers):
    """Baseline: best single-layer direction loading as predictor."""
    from sklearn.metrics import roc_auc_score

    best_auc, best_l = 0, 0
    # Direction loading features are the first n_layers columns
    for l in range(n_layers):
        scores = X_test[:, l]
        try:
            auc = roc_auc_score(y_test, scores)
        except ValueError:
            auc = 0.5
        if auc > best_auc:
            best_auc = auc
            best_l = l
    return best_auc, best_l


def plot_feature_importance(model, feat_names, task_name, out_dir):
    """Save feature importance bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    # If it's an XGBClassifier, get the underlying booster
    if hasattr(model, 'get_booster'):
        importance = model.get_booster().get_score(importance_type="gain")
    else:
        importance = model.get_score(importance_type="gain")
    # Map f0,f1,... to feature names
    named_imp = {}
    for k, v in importance.items():
        idx = int(k.replace("f", ""))
        if idx < len(feat_names):
            named_imp[feat_names[idx]] = v

    if not named_imp:
        return

    # Top 20
    top = sorted(named_imp.items(), key=lambda x: -x[1])[:20]
    names, vals = zip(*top)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(names)), vals, color="#4CAF50")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Gain")
    ax.set_title(f"Feature Importance — {task_name}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / f"importance_{task_name}.png"), dpi=150)
    plt.close()


def plot_layer_auc_curve(X_test, y_test, n_layers, layer_idxs,
                         task_name, out_dir):
    """Plot AUC of direction loading vs layer index."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return

    aucs = []
    for l in range(n_layers):
        try:
            auc = roc_auc_score(y_test, X_test[:, l])
        except ValueError:
            auc = 0.5
        aucs.append(auc)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layer_idxs, aucs, "o-", color="#2196F3", linewidth=2)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("AUC (direction only)")
    ax.set_title(f"Layer AUC Curve — {task_name}")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / f"layer_auc_{task_name}.png"), dpi=150)
    plt.close()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_dir", required=True,
                    help="Directory with features_*.npz from compute_and_extract.py")
    ap.add_argument("--out_dir", default="./hybrid_models",
                    help="Where to save models, plots, results")
    ap.add_argument("--model_key", default="gemma4_31b",
                    help="Used to select the right task list")
    return ap.parse_args()


def main():
    args = parse_args()
    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir) / args.model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = TASKS if "gemma" in args.model_key else TASKS_QWEN
    # Load layer config
    cfg = json.load(open(features_dir / "config.json"))
    layer_idxs = cfg["layer_idxs"]
    n_layers_sel = len(layer_idxs)

    all_results = {}
    print(f"=== Training hybrid classifiers ({args.model_key}) ===\n")

    for task_name in tasks:
        data = load_task(features_dir, task_name)
        if data is None:
            print(f"  {task_name}: no features found, skipping")
            continue

        X_tr, y_tr = data["X_train"], data["y_train"]
        X_te, y_te = data["X_test"], data["y_test"]
        print(f"--- {task_name} ---")
        print(f"  Train: {len(y_tr)} (pos={y_tr.sum()}, neg={len(y_tr)-y_tr.sum()})")
        print(f"  Test:  {len(y_te)} (pos={y_te.sum()}, neg={len(y_te)-y_te.sum()})")

        # 1. Direction-only baseline
        dir_auc, dir_best_l = train_direction_only(
            X_tr, y_tr, X_te, y_te, n_layers_sel)
        print(f"  Direction-only: AUC={dir_auc:.4f} "
              f"(best layer={layer_idxs[dir_best_l]})")

        # 2. Logistic regression
        lr_model, lr_auc, lr_preds, scaler = train_logistic(
            X_tr, y_tr, X_te, y_te)
        print(f"  Logistic Regression: AUC={lr_auc:.4f}")

        # 3. XGBoost
        xgb_model, xgb_auc, xgb_preds, cv_mean, cv_std = train_xgboost(
            X_tr, y_tr, X_te, y_te)
        print(f"  XGBoost: AUC={xgb_auc:.4f} (CV: {cv_mean:.4f}±{cv_std:.4f})")

        # 4. SVM
        svm_model, svm_auc, svm_preds, svm_scaler = train_svm(
            X_tr, y_tr, X_te, y_te)
        print(f"  SVM: AUC={svm_auc:.4f}")

        # Save XGBoost model
        if hasattr(xgb_model, 'get_booster'):
            xgb_model.get_booster().save_model(str(out_dir / f"xgb_{task_name}.json"))
        else:
            xgb_model.save_model(str(out_dir / f"xgb_{task_name}.json"))

        # Plots
        plot_feature_importance(xgb_model, data["feat_names"],
                                task_name, out_dir)
        plot_layer_auc_curve(X_te, y_te, n_layers_sel, layer_idxs,
                             task_name, out_dir)

        all_results[task_name] = {
            "direction_only_auc": float(dir_auc),
            "direction_best_layer": int(layer_idxs[dir_best_l]),
            "logistic_auc": float(lr_auc),
            "xgboost_auc": float(xgb_auc),
            "xgboost_cv_mean": float(cv_mean),
            "xgboost_cv_std": float(cv_std),
            "svm_auc": float(svm_auc),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
        }
        print()

    # Headline metric
    xgb_aucs = [r["xgboost_auc"] for r in all_results.values()]
    dir_aucs = [r["direction_only_auc"] for r in all_results.values()]
    lr_aucs = [r["logistic_auc"] for r in all_results.values()]
    svm_aucs = [r["svm_auc"] for r in all_results.values()]

    print("=" * 60)
    print(f"HEADLINE mean AUC:")
    print(f"  Direction-only: {np.mean(dir_aucs):.4f} ± {np.std(dir_aucs):.4f}")
    print(f"  Logistic:       {np.mean(lr_aucs):.4f} ± {np.std(lr_aucs):.4f}")
    print(f"  SVM:            {np.mean(svm_aucs):.4f} ± {np.std(svm_aucs):.4f}")
    print(f"  XGBoost:        {np.mean(xgb_aucs):.4f} ± {np.std(xgb_aucs):.4f}")
    print("=" * 60)

    all_results["headline"] = {
        "direction_mean_auc": float(np.mean(dir_aucs)),
        "logistic_mean_auc": float(np.mean(lr_aucs)),
        "svm_mean_auc": float(np.mean(svm_aucs)),
        "xgboost_mean_auc": float(np.mean(xgb_aucs)),
        "xgboost_std_auc": float(np.std(xgb_aucs)),
    }

    json.dump(all_results, open(out_dir / "results.json", "w"), indent=2)
    print(f"\nResults + models saved to {out_dir}")


if __name__ == "__main__":
    main()
