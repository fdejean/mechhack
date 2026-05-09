"""Residual loader for score_probes.py.

Routes sample_id + dataset metadata to the correct extract .pt file,
loads it, and returns residuals + attention_mask as numpy arrays.

Dataset dispatch logic
----------------------
cyber_probes/test.jsonl:
  sample_id like "cyber_0" → extracts/cyber_gemma4_31b/cyber_0.pt
  (all cyber samples are gemma4_31b extractions)

refusal_probes/gemma4_31b/test_split.jsonl:
  sample_id like "EVO_0001" → extracts/gemma4_31b/EVO_0001.pt

refusal_probes/qwen36/test_split.jsonl:
  sample_id like "EVO_0001" → extracts/qwen36/EVO_0001.pt

The dispatch is based on the presence of "category" field (cyber) or
"model_key" field (refusal), falling back to sample_id prefix analysis.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

EXTRACT_ROOT = Path(__file__).resolve().parent.parent / "extracts"

# Mapping of model_key → extract sub-directory
_MODEL_EXTRACT_DIR = {
    "gemma4_31b": "gemma4_31b",
    "qwen36":     "qwen36",
}

# Cyber samples always come from gemma4_31b extractions
CYBER_EXTRACT_DIR = "cyber_gemma4_31b"


def _resolve_extract_path(row: dict[str, Any]) -> Path | None:
    """Return the Path to the extract .pt file for a JSONL row, or None if not found."""

    sample_id: str = row.get("sample_id", "")
    category: str | None = row.get("category")          # cyber_probes
    model_key: str | None = row.get("model_key")        # refusal_probes

    # ── Cyber probes ──────────────────────────────────────────────────────
    if category is not None:
        # sample_id like "cyber_0" → extracts/cyber_gemma4_31b/cyber_0.pt
        path = EXTRACT_ROOT / CYBER_EXTRACT_DIR / f"{sample_id}.pt"
        if path.exists():
            return path
        # Try without .pt extension in sample_id (sample_id might already be bare)
        path_no_ext = EXTRACT_ROOT / CYBER_EXTRACT_DIR / sample_id
        for ext in (".pt",):
            if (path_no_ext.parent / (path_no_ext.name + ext)).exists():
                return path_no_ext.parent / (path_no_ext.name + ext)
        return path  # Return expected path even if missing — caller handles

    # ── Refusal probes ────────────────────────────────────────────────────
    if model_key is not None:
        extract_subdir = _MODEL_EXTRACT_DIR.get(model_key)
        if extract_subdir is None:
            logger.warning("Unknown model_key %r for sample %r", model_key, sample_id)
            return None
        path = EXTRACT_ROOT / extract_subdir / f"{sample_id}.pt"
        if path.exists():
            return path
        return path  # Return expected path — caller handles

    # ── Fallback: infer from sample_id prefix ─────────────────────────────
    if sample_id.startswith("cyber_"):
        return EXTRACT_ROOT / CYBER_EXTRACT_DIR / f"{sample_id}.pt"

    if sample_id.startswith("EVO_"):
        # Can't distinguish gemma vs qwen without model_key — try gemma first
        for subdir in ("gemma4_31b", "qwen36"):
            p = EXTRACT_ROOT / subdir / f"{sample_id}.pt"
            if p.exists():
                return p
        # Return gemma path as default
        return EXTRACT_ROOT / "gemma4_31b" / f"{sample_id}.pt"

    logger.warning("Cannot route sample_id %r — no category or model_key field", sample_id)
    return None


def residual_loader(row: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """Load residuals and attention_mask for a single JSONL row.

    Args:
        row: A dict parsed from one line of cyber_probes/test.jsonl or
             refusal_probes/{gemma4_31b,qwen36}/test_split.jsonl.
             Must contain at least ``sample_id``; ``category`` (cyber) or
             ``model_key`` (refusal) is used for routing.

    Returns:
        {"residuals": np.ndarray, "attention_mask": np.ndarray}
        residuals shape: (n_selected_layers, n_tokens, d_model), dtype float16
        attention_mask shape: (n_tokens,), dtype bool

        Returns None if the file is not found (caller should skip the row).
    Raises:
        FileNotFoundError: if the expected .pt path does not exist.
        KeyError: if required fields (sample_id) are missing.
    """
    sample_id = row.get("sample_id")
    if not sample_id:
        raise KeyError("Row missing 'sample_id' field")

    path = _resolve_extract_path(row)
    if path is None:
        logger.error("Cannot resolve extract path for sample_id=%r", sample_id)
        return None

    if not path.exists():
        logger.warning("Extract not found: %s  (sample_id=%r)", path, sample_id)
        return None

    try:
        data = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return None

    residuals: torch.Tensor = data["residuals"]
    attention_mask_raw = data["attention_mask"]

    # Convert torch tensor or list-of-bool to numpy
    if isinstance(attention_mask_raw, torch.Tensor):
        attention_mask = attention_mask_raw.numpy().astype(bool)
    else:
        attention_mask = np.asarray(attention_mask_raw, dtype=bool)

    # Ensure residuals is numpy (copy to avoid aliasing torch fp16)
    residuals_np = residuals.numpy()

    return {
        "residuals":     residuals_np,
        "attention_mask": attention_mask,
    }