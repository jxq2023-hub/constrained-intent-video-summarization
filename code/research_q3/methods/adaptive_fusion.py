"""Confidence-aware visual-semantic score fusion.

Lower normalized entropy means that a modality makes a more selective,
confident prediction. The confidence-derived weight is blended with an
intent-conditioned semantic prior so that uncertainty and user intent both
affect the final score.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    span = float(values.max() - values.min()) if values.size else 0.0
    if span <= 1e-12:
        return np.zeros_like(values)
    return (values - values.min()) / span


def _entropy_confidence(normalized_scores: np.ndarray) -> tuple[float, float]:
    n_items = normalized_scores.size
    if n_items <= 1 or float(normalized_scores.sum()) <= 1e-12:
        return 1.0, 0.0
    probabilities = normalized_scores / normalized_scores.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
    normalized_entropy = float(np.clip(entropy / np.log(n_items), 0.0, 1.0))
    return normalized_entropy, 1.0 - normalized_entropy


def confidence_adaptive_fusion(
    visual_scores: np.ndarray,
    semantic_scores: np.ndarray,
    semantic_prior: float,
    prior_strength: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fuse visual and semantic evidence using confidence and intent prior."""
    visual = _minmax(np.asarray(visual_scores))
    semantic = _minmax(np.asarray(semantic_scores))
    if visual.shape != semantic.shape:
        raise ValueError("visual_scores and semantic_scores must have identical shape")
    semantic_prior = float(np.clip(semantic_prior, 0.0, 1.0))
    prior_strength = float(np.clip(prior_strength, 0.0, 1.0))

    visual_entropy, visual_confidence = _entropy_confidence(visual)
    semantic_entropy, semantic_confidence = _entropy_confidence(semantic)
    confidence_sum = visual_confidence + semantic_confidence
    if confidence_sum <= 1e-12:
        confidence_weight = semantic_prior
    else:
        confidence_weight = semantic_confidence / confidence_sum

    semantic_weight = (
        prior_strength * semantic_prior
        + (1.0 - prior_strength) * confidence_weight
    )
    fused = semantic_weight * semantic + (1.0 - semantic_weight) * visual
    diagnostics = {
        "visual_entropy_normalized": visual_entropy,
        "semantic_entropy_normalized": semantic_entropy,
        "visual_confidence": visual_confidence,
        "semantic_confidence": semantic_confidence,
        "confidence_semantic_weight": float(confidence_weight),
        "semantic_prior": semantic_prior,
        "prior_strength": prior_strength,
        "semantic_weight": float(semantic_weight),
    }
    return fused, diagnostics

