"""Nested leave-one-video-out selection for QFVS scoring rules.

Every held-out video's scoring configuration is selected only from the other
three videos.  The predeclared selection objective is the equal-weight mean of
tag-grounded either-concept and both-concepts relevance.  The expensive
official semantic matching is computed only for the final held-out results.
"""

from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from research_q3.methods.diverse_selection import mmr_select
from research_q3.protocols.qfvs_semantic import evaluate_exact_and_query_relevance, evaluate_semantic_summary, load_qfvs_tags


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research_q3" / "data" / "qfvs"
CACHE = BASE / "feature_cache"
ANNOTATIONS = BASE / "annotations"
CONCEPTS = ROOT / "research_q3" / "configs" / "qfvs_concepts.json"
OUT = ROOT / "research_q3" / "results" / "qfvs_nested_selection_results.csv"
SWEEP_OUT = ROOT / "research_q3" / "results" / "qfvs_nested_selection_sweep.csv"


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    return (values - values.min()) / span if span > 1e-12 else np.zeros_like(values)


def combine(a: np.ndarray, b: np.ndarray, rule: str) -> np.ndarray:
    a, b = minmax(a), minmax(b)
    if rule == "mean":
        return 0.5 * (a + b)
    if rule == "max":
        return np.maximum(a, b)
    if rule == "min":
        return np.minimum(a, b)
    if rule == "geometric":
        return np.sqrt(np.maximum(a, 0) * np.maximum(b, 0))
    raise ValueError(rule)


def query_files(video: str) -> list[Path]:
    return sorted((ANNOTATIONS / "Oracle_summaries" / video).glob("*_oracle.txt"))


def parse_query(path: Path) -> tuple[str, str]:
    fields = path.stem.removesuffix("_oracle").split("_")
    return fields[0], fields[1]


def configuration_name(rule: str, visual_weight: float, diversity: float) -> str:
    return f"rule={rule};visual={visual_weight:.2f};diversity={diversity:.2f}"


def main() -> None:
    concept_names = list(json.loads(CONCEPTS.read_text("utf-8")))
    tags = load_qfvs_tags(str(ANNOTATIONS / "Tags.mat"))
    videos = ["P01", "P02", "P03", "P04"]
    cached = {}
    for video in videos:
        base = np.load(CACHE / f"{video}.npz")
        cached[video] = {
            "visual": minmax(base["visual_scores"]),
            "image": np.load(CACHE / f"{video}_openai_clip.npz")["image_features"].astype(np.float32),
        }

    calibrated = {}
    for video in videos:
        train = [name for name in videos if name != video]
        train_x = np.concatenate([cached[name]["image"] for name in train])
        train_y = np.concatenate([tags[int(name[-1]) - 1] for name in train]).astype(np.float32)
        head = Ridge(alpha=10.0, fit_intercept=True, solver="lsqr").fit(train_x, train_y)
        calibrated[video] = head.predict(cached[video]["image"])

    configs = list(product(
        ["mean", "max", "min", "geometric"],
        [0.0, 0.10, 0.20],
        [0.0, 0.10, 0.20],
    ))
    sweep_rows = []
    predictions: dict[tuple[str, str, str], dict] = {}
    for video in videos:
        video_index = int(video[-1]) - 1
        image = cached[video]["image"]
        visual = cached[video]["visual"]
        k = max(1, int(len(visual) * 0.02))
        for path in query_files(video):
            concept1, concept2 = parse_query(path)
            c1, c2 = concept_names.index(concept1), concept_names.index(concept2)
            oracle = [int(line.strip()) - 1 for line in path.read_text("utf-8").splitlines() if line.strip()]
            for rule, visual_weight, diversity in configs:
                semantic = combine(calibrated[video][:, c1], calibrated[video][:, c2], rule)
                score = (1.0 - visual_weight) * semantic + visual_weight * visual
                selected = (
                    mmr_select(score, image, k, diversity=diversity)
                    if diversity > 0 else np.argsort(-score, kind="stable")[:k]
                )
                auxiliary = evaluate_exact_and_query_relevance(selected, oracle, tags[video_index], c1, c2)
                config = configuration_name(rule, visual_weight, diversity)
                row = {
                    "video": video, "query_id": path.stem.removesuffix("_oracle"),
                    "config": config, **auxiliary,
                    "selection_objective": 0.5 * auxiliary["query_relevance_or"] + 0.5 * auxiliary["query_relevance_and"],
                }
                sweep_rows.append(row)
                predictions[(video, row["query_id"], config)] = {
                    **row, "selected_indices": json.dumps([int(value) for value in selected]),
                }
        print(f"swept {video}", flush=True)

    sweep = __import__("pandas").DataFrame(sweep_rows)
    final_rows = []
    for test_video in videos:
        training = sweep[sweep.video != test_video]
        rankings = training.groupby("config").selection_objective.mean().sort_values(ascending=False)
        chosen = rankings.index[0]
        for path in query_files(test_video):
            query_id = path.stem.removesuffix("_oracle")
            prediction = predictions[(test_video, query_id, chosen)]
            oracle = [int(line.strip()) - 1 for line in path.read_text("utf-8").splitlines() if line.strip()]
            selected = json.loads(prediction["selected_indices"])
            official = evaluate_semantic_summary(
                selected, oracle, tags[int(test_video[-1]) - 1]
            )
            final_rows.append({
                **prediction, "fscore": official["fscore"],
                "method": "Ours-Nested-Intent-Selector", "held_out_video": test_video,
                "training_objective": float(rankings.iloc[0]),
            })
        print(f"held-out {test_video}: {chosen} train_objective={rankings.iloc[0]:.4f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in ((SWEEP_OUT, sweep_rows), (OUT, final_rows)):
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    final = __import__("pandas").DataFrame(final_rows)
    print(final[["fscore", "exact_fscore", "query_relevance_or", "query_relevance_and"]].mean())


if __name__ == "__main__":
    main()
