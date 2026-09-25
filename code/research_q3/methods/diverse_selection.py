"""Deterministic relevance-diversity selection for query-focused summaries."""

from __future__ import annotations

import numpy as np


def mmr_select(scores: np.ndarray, embeddings: np.ndarray, k: int, diversity: float = 0.15,
               shortlist_factor: int = 10) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    features = np.asarray(embeddings, dtype=float)
    k = min(max(int(k), 0), len(scores))
    if k == 0:
        return np.asarray([], dtype=np.int64)
    span = scores.max() - scores.min()
    relevance = (scores - scores.min()) / span if span > 1e-12 else np.zeros_like(scores)
    pool_size = min(len(scores), max(k, k * int(shortlist_factor)))
    pool = np.argsort(-relevance, kind="stable")[:pool_size]
    vectors = features[pool]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    chosen_local: list[int] = []
    max_similarity = np.zeros(pool_size, dtype=float)
    available = np.ones(pool_size, dtype=bool)
    for _ in range(k):
        objective = (1.0 - diversity) * relevance[pool] - diversity * max_similarity
        objective[~available] = -np.inf
        index = int(np.argmax(objective))
        chosen_local.append(index)
        available[index] = False
        similarity = vectors @ vectors[index]
        max_similarity = np.maximum(max_similarity, similarity)
    return np.sort(pool[np.asarray(chosen_local, dtype=int)]).astype(np.int64)
