"""Standard SumMe/TVSum summary generation and F-score evaluation.

The protocol follows the commonly used evaluation pipeline: sampled frame
scores are expanded to the original frame rate, shot scores are averaged over
KTS segments, shots are selected under a 15% duration budget by 0/1 knapsack,
and the resulting binary summary is compared with every user summary. TVSum
uses the average user F-score; SumMe uses the maximum user F-score.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np


def expand_sample_scores(
    sample_scores: Iterable[float], picks: Iterable[int], n_frames: int
) -> np.ndarray:
    """Expand subsampled scores to a dense frame-level score sequence."""
    scores = np.asarray(list(sample_scores), dtype=np.float64).reshape(-1)
    picks_arr = np.asarray(list(picks), dtype=np.int64).reshape(-1)
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if scores.size == 0 or picks_arr.size == 0:
        return np.zeros(n_frames, dtype=np.float64)
    if scores.size != picks_arr.size:
        raise ValueError("sample_scores and picks must have the same length")

    dense = np.zeros(n_frames, dtype=np.float64)
    starts = np.clip(picks_arr, 0, n_frames - 1)
    ends = np.r_[starts[1:], n_frames]
    for score, start, end in zip(scores, starts, ends):
        dense[int(start) : max(int(start) + 1, int(end))] = float(score)
    return dense


def _knapsack(values: np.ndarray, weights: np.ndarray, capacity: int) -> list[int]:
    """Deterministic 0/1 knapsack returning selected item indices."""
    n_items = len(values)
    if n_items == 0 or capacity <= 0:
        return []
    dp = np.zeros((n_items + 1, capacity + 1), dtype=np.float64)
    keep = np.zeros((n_items + 1, capacity + 1), dtype=bool)
    for i in range(1, n_items + 1):
        weight = int(weights[i - 1])
        value = float(values[i - 1])
        dp[i] = dp[i - 1]
        if weight <= capacity:
            candidates = dp[i - 1, : capacity + 1 - weight] + value
            better = candidates > dp[i, weight:] + 1e-12
            dp[i, weight:][better] = candidates[better]
            keep[i, weight:][better] = True

    selected: list[int] = []
    remaining = int(np.argmax(dp[n_items]))
    for i in range(n_items, 0, -1):
        if keep[i, remaining]:
            selected.append(i - 1)
            remaining -= int(weights[i - 1])
    return selected[::-1]


def generate_summary(
    frame_scores: Iterable[float],
    change_points: np.ndarray,
    n_frames: int,
    budget_ratio: float = 0.15,
) -> np.ndarray:
    """Generate a binary video summary using shot means and knapsack."""
    scores = np.asarray(list(frame_scores), dtype=np.float64).reshape(-1)
    if scores.size != n_frames:
        raise ValueError(f"expected {n_frames} frame scores, got {scores.size}")
    cps = np.asarray(change_points, dtype=np.int64).reshape(-1, 2)
    if cps.size == 0:
        return np.zeros(n_frames, dtype=np.int8)

    shot_lengths = cps[:, 1] - cps[:, 0] + 1
    shot_values = np.asarray(
        [scores[max(0, s) : min(n_frames, e + 1)].mean() for s, e in cps],
        dtype=np.float64,
    )
    capacity = max(1, int(np.floor(n_frames * budget_ratio)))
    selected = _knapsack(shot_values, shot_lengths, capacity)

    summary = np.zeros(n_frames, dtype=np.int8)
    for index in selected:
        start, end = cps[index]
        summary[max(0, start) : min(n_frames, end + 1)] = 1
    return summary


def evaluate_fscore(
    predicted_summary: Iterable[int],
    user_summaries: np.ndarray,
    method: Literal["avg", "max"],
) -> dict[str, float | list[float]]:
    """Evaluate a binary summary against all human summaries."""
    pred = np.asarray(list(predicted_summary), dtype=bool).reshape(-1)
    users = np.asarray(user_summaries, dtype=bool)
    if users.ndim != 2:
        raise ValueError("user_summaries must be a 2-D array")
    if method not in {"avg", "max"}:
        raise ValueError("method must be 'avg' or 'max'")

    length = max(pred.size, users.shape[1])
    pred_pad = np.zeros(length, dtype=bool)
    pred_pad[: pred.size] = pred
    per_user = []
    precisions = []
    recalls = []
    for user in users:
        gt = np.zeros(length, dtype=bool)
        gt[: user.size] = user
        overlap = np.logical_and(pred_pad, gt).sum()
        precision = float(overlap / pred_pad.sum()) if pred_pad.sum() else 0.0
        recall = float(overlap / gt.sum()) if gt.sum() else 0.0
        fscore = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        per_user.append(fscore)

    aggregate = float(np.mean(per_user) if method == "avg" else np.max(per_user))
    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "fscore": aggregate,
        "per_user_fscore": [float(value) for value in per_user],
    }

