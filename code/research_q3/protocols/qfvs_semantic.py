"""Semantic matching protocol for the QFVS benchmark."""

from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np
import scipy.io


def load_qfvs_tags(path: str) -> list[np.ndarray]:
    """Load the official Tags.mat into one shot-by-concept array per video."""
    raw_videos = scipy.io.loadmat(path)["Tags"][0]
    videos = []
    for raw_video in raw_videos:
        shots = []
        for raw_shot in raw_video[0]:
            shots.append(np.asarray(raw_shot[0][0]).reshape(-1))
        videos.append(np.asarray(shots))
    return videos


def _semantic_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = np.asarray(a, dtype=bool)
    b_bool = np.asarray(b, dtype=bool)
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a_bool, b_bool).sum() / union)


def evaluate_semantic_summary(
    predicted_shots: Iterable[int],
    oracle_shots: Iterable[int],
    dense_shot_tags: np.ndarray,
) -> dict[str, float]:
    """Maximum-weight semantic matching between machine and oracle shots."""
    pred = np.asarray(sorted(set(int(i) for i in predicted_shots)), dtype=np.int64)
    oracle = np.asarray(sorted(set(int(i) for i in oracle_shots)), dtype=np.int64)
    tags = np.asarray(dense_shot_tags)
    if pred.size == 0 or oracle.size == 0:
        return {"precision": 0.0, "recall": 0.0, "fscore": 0.0}
    if pred.min() < 0 or oracle.min() < 0 or pred.max() >= len(tags) or oracle.max() >= len(tags):
        raise IndexError("shot index is outside dense_shot_tags")

    graph = nx.Graph()
    for i, pred_index in enumerate(pred):
        for j, oracle_index in enumerate(oracle):
            graph.add_edge(
                f"pred:{i}",
                f"oracle:{j}",
                weight=_semantic_iou(tags[pred_index], tags[oracle_index]),
            )
    matching = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=False)
    matched_similarity = sum(graph.get_edge_data(a, b)["weight"] for a, b in matching)
    precision = float(matched_similarity / pred.size)
    recall = float(matched_similarity / oracle.size)
    fscore = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "fscore": float(fscore)}


def evaluate_exact_and_query_relevance(
    predicted_shots: Iterable[int], oracle_shots: Iterable[int], dense_shot_tags: np.ndarray,
    concept1_index: int, concept2_index: int,
) -> dict[str, float]:
    """Complement official semantic F1 with exact overlap and tag relevance."""
    pred = np.asarray(sorted(set(int(i) for i in predicted_shots)), dtype=np.int64)
    oracle = np.asarray(sorted(set(int(i) for i in oracle_shots)), dtype=np.int64)
    overlap = len(np.intersect1d(pred, oracle))
    precision = overlap / len(pred) if len(pred) else 0.0
    recall = overlap / len(oracle) if len(oracle) else 0.0
    fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if len(pred):
        selected = np.asarray(dense_shot_tags)[pred]
        first = selected[:, concept1_index] > 0
        second = selected[:, concept2_index] > 0
        relevance_or = float(np.logical_or(first, second).mean())
        relevance_and = float(np.logical_and(first, second).mean())
    else:
        relevance_or = relevance_and = 0.0
    return {
        "exact_precision": float(precision), "exact_recall": float(recall),
        "exact_fscore": float(fscore), "query_relevance_or": relevance_or,
        "query_relevance_and": relevance_and,
    }
