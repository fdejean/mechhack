"""Scoring interface for the hybrid predictor.

Conforms to the predict.py spec in rules/predict.md:
  load_predictor() -> predictor object
  predict(predictor, residuals, attention_mask) -> float in [0,1]

The predictor loads:
  - Per-task XGBoost models
  - Per-task direction vectors
  - W_U (unembedding matrix) for logit-lens features
  - Refusal/compliance token IDs

Place this file in your submission directory alongside the hybrid_models/
and hybrid_artifacts/ folders.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Paths relative to this file — adjust if needed
BASE = Path(__file__).resolve().parent

# Default task (scoring infra passes context separately)
# The scoring script calls predict() with residuals — it doesn't tell us
# which task. We detect from the call pattern in score_probes.py.
_CURRENT_TASK = None


def load_predictor(
    artifacts_dir=None,
    models_dir=None,
    model_key="gemma4_31b",
):
    """Load all artifacts needed for prediction.

    Returns an opaque predictor dict.
    """
    import xgboost as xgb

    if artifacts_dir is None:
        artifacts_dir = BASE / "hybrid_artifacts" / model_key
    if models_dir is None:
        models_dir = BASE / "hybrid_models" / model_key
    artifacts_dir = Path(artifacts_dir)
    models_dir = Path(models_dir)

    # Config
    cfg = json.load(open(artifacts_dir / "config.json"))
    layer_idxs = cfg["layer_idxs"]

    # W_U for logit lens
    W_U = torch.load(str(artifacts_dir / "W_U.pt"),
                      weights_only=True, map_location="cpu")

    # Token IDs
    refusal_ids = json.load(
        open(artifacts_dir / "refusal_token_ids.json"))["ids"]
    compliance_ids = json.load(
        open(artifacts_dir / "compliance_token_ids.json"))["ids"]

    # Per-task: XGBoost model + direction
    tasks = {}
    for task_name in ["refusal_gemma4_31b", "refusal_qwen36",
                      "cyber_probe1", "cyber_probe2", "cyber_probe3"]:
        xgb_path = models_dir / f"xgb_{task_name}.json"
        dir_path = artifacts_dir / f"direction_{task_name}.pt"
        if not xgb_path.exists() or not dir_path.exists():
            continue
        booster = xgb.Booster()
        booster.load_model(str(xgb_path))
        d = torch.load(str(dir_path), weights_only=False, map_location="cpu")
        tasks[task_name] = {
            "booster": booster,
            "direction": d["direction"],  # (n_layers_sel, d_model)
        }

    return {
        "tasks": tasks,
        "W_U": W_U,
        "refusal_ids": refusal_ids,
        "compliance_ids": compliance_ids,
        "layer_idxs": layer_idxs,
        "model_key": model_key,
    }


def set_task(task_name):
    """Set which task the next predict() calls are for."""
    global _CURRENT_TASK
    _CURRENT_TASK = task_name


def predict(predictor, residuals, attention_mask=None):
    """Predict probability of positive class.

    Args:
        predictor: from load_predictor()
        residuals: np.ndarray (n_layers_sel, n_tokens, d_model) fp16
        attention_mask: np.ndarray (n_tokens,) bool, optional

    Returns:
        float in [0, 1]
    """
    import xgboost as xgb

    global _CURRENT_TASK
    task_name = _CURRENT_TASK
    if task_name is None or task_name not in predictor["tasks"]:
        # Fallback: use the first available task
        task_name = list(predictor["tasks"].keys())[0]

    task = predictor["tasks"][task_name]
    direction = task["direction"]  # (n_layers_sel, d_model)
    booster = task["booster"]

    # Convert to torch tensors
    res = torch.from_numpy(np.asarray(residuals)).float()  # (n_layers, n_tok, d)
    W_U = predictor["W_U"]
    refusal_ids = predictor["refusal_ids"]
    compliance_ids = predictor["compliance_ids"]

    # Find last real token
    if attention_mask is not None:
        mask = np.asarray(attention_mask).astype(bool)
        last_idx = int(np.where(mask)[0].max()) if mask.any() else 0
    else:
        last_idx = res.shape[1] - 1

    last_tok = res[:, last_idx, :]  # (n_layers, d_model)
    n_layers = last_tok.shape[0]

    # Direction loading per layer
    dir_loading = (last_tok * direction).sum(dim=-1).numpy()  # (n_layers,)

    # Logit lens
    logits = last_tok @ W_U.T  # (n_layers, vocab_size)
    probs = F.softmax(logits, dim=-1)
    ref_p = probs[:, refusal_ids].sum(dim=-1).numpy()
    comp_p = probs[:, compliance_ids].sum(dim=-1).numpy()

    # Derived features
    dir_diff = np.diff(dir_loading)
    transition = float(np.argmax(np.abs(dir_diff))) if len(dir_diff) > 0 else 0
    peak = float(np.argmax(dir_loading))
    total = float(dir_loading.sum())
    max_ratio = float(np.max(
        np.log(ref_p.clip(1e-10) / comp_p.clip(1e-10))))
    n_tokens = float(res.shape[1])

    # Assemble feature vector (must match train_hybrid.py order)
    feats = np.concatenate([
        dir_loading, ref_p, comp_p,
        [transition, peak, total, max_ratio, n_tokens],
    ]).reshape(1, -1).astype(np.float32)

    dmat = xgb.DMatrix(feats)
    prob = float(booster.predict(dmat)[0])
    return prob
