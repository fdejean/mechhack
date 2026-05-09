"""Embedding distance tracker for Level 2 attacks.

Computes distance metrics between original and edited prompts:
  1. Input embedding cosine similarity (cheap — no forward pass)
  2. L2 distance in input embedding space
  3. Direction-loading delta (how much the refusal signal moved)
  4. Full contextual distance (last hidden state — expensive, batch mode)
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F
import numpy as np


def _mean_input_embedding(model, tokenizer, text: str,
                          device: str = "cuda:0") -> torch.Tensor:
    """Compute mean of input embeddings for a text (no forward pass needed)."""
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=2048).to(device)
    embed_layer = model.get_input_embeddings()
    with torch.no_grad():
        embeddings = embed_layer(enc.input_ids)  # (1, seq_len, d_emb)
        # Mean pool over non-padding tokens
        mask = enc.attention_mask.unsqueeze(-1).float()  # (1, seq_len, 1)
        mean_emb = (embeddings * mask).sum(dim=1) / mask.sum(dim=1)  # (1, d_emb)
    return mean_emb.squeeze(0).float()  # (d_emb,)


def input_embedding_distance(model, tokenizer,
                              original: str, edited: str,
                              device: str = "cuda:0") -> dict:
    """Compute distance between original and edited in input embedding space.

    This is CHEAP — only uses the embedding lookup table, no forward pass.

    Returns:
        {cosine_similarity, cosine_distance, l2_distance}
    """
    emb_orig = _mean_input_embedding(model, tokenizer, original, device)
    emb_edit = _mean_input_embedding(model, tokenizer, edited, device)

    cos_sim = F.cosine_similarity(emb_orig.unsqueeze(0),
                                   emb_edit.unsqueeze(0)).item()
    l2 = torch.norm(emb_orig - emb_edit).item()

    return {
        "cosine_similarity": round(cos_sim, 6),
        "cosine_distance": round(1.0 - cos_sim, 6),
        "l2_distance": round(l2, 4),
    }


def direction_loading_delta(residuals_orig: torch.Tensor,
                             residuals_edit: torch.Tensor,
                             direction: torch.Tensor,
                             layer_weights: torch.Tensor) -> dict:
    """Compute the change in refusal-direction loading between orig and edited.

    This is essentially FREE if you already have both sets of residuals.

    Args:
        residuals_orig: (n_layers, n_tokens_orig, d_model)
        residuals_edit: (n_layers, n_tokens_edit, d_model)
        direction: (n_layers, d_model) unit vectors
        layer_weights: (n_layers,) importance weights from classifier

    Returns:
        {loading_orig, loading_edit, loading_delta, loading_pct_change}
    """
    def _mean_loading(residuals: torch.Tensor) -> float:
        # Per-token per-layer loading → mean across tokens → weighted sum across layers
        loadings = torch.einsum("ltd,ld->lt", residuals.float(),
                                direction.float())  # (n_layers, n_tokens)
        mean_per_layer = loadings.mean(dim=1)  # (n_layers,)
        weighted = (mean_per_layer * layer_weights.float()).sum().item()
        return weighted

    loading_orig = _mean_loading(residuals_orig)
    loading_edit = _mean_loading(residuals_edit)
    delta = loading_edit - loading_orig
    pct = (delta / abs(loading_orig) * 100) if abs(loading_orig) > 1e-8 else 0.0

    return {
        "loading_orig": round(loading_orig, 6),
        "loading_edit": round(loading_edit, 6),
        "loading_delta": round(delta, 6),
        "loading_pct_change": round(pct, 2),
    }


def full_contextual_distance(model, tokenizer,
                              original: str, edited: str,
                              device: str = "cuda:0",
                              target_layer: int = -1) -> dict:
    """Compute distance using full forward-pass hidden states.

    EXPENSIVE — requires a forward pass for each text. Use only for
    final successful edits, not in the search loop.

    Args:
        target_layer: which hidden state layer to use. -1 = last.

    Returns:
        {cosine_similarity, cosine_distance, l2_distance, layer_used}
    """
    def _get_hidden(text: str) -> torch.Tensor:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=2048).to(device)
        with torch.no_grad():
            out = model(input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask,
                        output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[target_layer][0]  # (seq_len, d_model)
        mask = enc.attention_mask[0].unsqueeze(-1).float()
        mean_h = (hs.float() * mask).sum(dim=0) / mask.sum(dim=0)
        del out
        torch.cuda.empty_cache()
        return mean_h

    h_orig = _get_hidden(original)
    h_edit = _get_hidden(edited)

    cos_sim = F.cosine_similarity(h_orig.unsqueeze(0),
                                   h_edit.unsqueeze(0)).item()
    l2 = torch.norm(h_orig - h_edit).item()

    n_layers = len(model.model.layers) if hasattr(model.model, "layers") else "unknown"
    actual_layer = target_layer if target_layer >= 0 else (
        n_layers + target_layer if isinstance(n_layers, int) else target_layer)

    return {
        "cosine_similarity": round(cos_sim, 6),
        "cosine_distance": round(1.0 - cos_sim, 6),
        "l2_distance": round(l2, 4),
        "layer_used": actual_layer,
    }


def compute_all_distances(model, tokenizer,
                           original: str, edited: str,
                           direction: Optional[torch.Tensor] = None,
                           layer_weights: Optional[torch.Tensor] = None,
                           residuals_orig: Optional[torch.Tensor] = None,
                           residuals_edit: Optional[torch.Tensor] = None,
                           device: str = "cuda:0",
                           include_contextual: bool = False) -> dict:
    """Compute all available distance metrics.

    Always computes input embedding distance (cheap).
    Computes direction loading delta if residuals + direction provided.
    Computes full contextual distance only if include_contextual=True.
    """
    result = {}

    # 1. Input embedding distance (always — it's cheap)
    result["input_embedding"] = input_embedding_distance(
        model, tokenizer, original, edited, device)

    # 2. Direction loading delta (if we have the ingredients)
    if (direction is not None and layer_weights is not None
            and residuals_orig is not None and residuals_edit is not None):
        result["direction_loading"] = direction_loading_delta(
            residuals_orig, residuals_edit, direction, layer_weights)

    # 3. Full contextual distance (expensive — opt-in)
    if include_contextual:
        result["contextual"] = full_contextual_distance(
            model, tokenizer, original, edited, device)

    return result
