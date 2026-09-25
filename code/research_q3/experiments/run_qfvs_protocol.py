"""Run official semantic-matching QFVS experiments on 4 videos × 45 queries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
import clip
import torch

from research_q3.features.sampled_video_scorer import SampledVideoScorer
from research_q3.methods.adaptive_fusion import confidence_adaptive_fusion
from research_q3.methods.diverse_selection import mmr_select
from research_q3.protocols.qfvs_semantic import (
    evaluate_exact_and_query_relevance, evaluate_semantic_summary, load_qfvs_tags,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research_q3" / "data" / "qfvs"
ANNOTATIONS = BASE / "annotations"
CACHE = BASE / "feature_cache"
CONCEPTS = ROOT / "research_q3" / "configs" / "qfvs_concepts.json"
OUT = ROOT / "research_q3" / "results" / "qfvs_results.csv"


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float); span = x.max() - x.min()
    return (x - x.min()) / span if span > 1e-12 else np.zeros_like(x)


def oracle_files(video: str) -> list[Path]:
    return sorted((ANNOTATIONS / "Oracle_summaries" / video).glob("*_oracle.txt"))


def parse_query(path: Path) -> tuple[str, str]:
    fields = path.stem.removesuffix("_oracle").split("_")
    if len(fields) != 2:
        raise ValueError(f"unexpected QFVS query filename: {path.name}")
    return fields[0], fields[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", choices=["P01", "P02", "P03", "P04", "all"], default="all")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    videos = ["P01", "P02", "P03", "P04"] if args.video == "all" else [args.video]
    concept_zh = json.loads(CONCEPTS.read_text("utf-8"))
    tags = load_qfvs_tags(str(ANNOTATIONS / "Tags.mat"))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    english_clip, _ = clip.load(str(ROOT / "models" / "ViT-B-32.pt"), device=device, jit=False)
    english_clip.eval()
    rows: list[dict] = []

    # Fit a concept calibration head under strict leave-one-video-out.  The
    # official 48-dimensional dense tags from the held-out video are never
    # used for fitting.
    cached = {}
    for video in ["P01", "P02", "P03", "P04"]:
        data = np.load(CACHE / f"{video}.npz")
        cached[video] = {
            "visual": minmax(data["visual_scores"]),
            "image": np.load(CACHE / f"{video}_openai_clip.npz")["image_features"].astype(np.float32),
        }

    for video in videos:
        video_index = int(video[-1]) - 1
        visual = cached[video]["visual"]
        image_features = cached[video]["image"]
        train_videos = [name for name in cached if name != video]
        train_x = np.concatenate([cached[name]["image"] for name in train_videos], axis=0)
        train_y = np.concatenate([tags[int(name[-1]) - 1] for name in train_videos], axis=0).astype(np.float32)
        concept_head = Ridge(alpha=10.0, fit_intercept=True, solver="lsqr")
        concept_head.fit(train_x, train_y)
        calibrated_concepts = concept_head.predict(image_features)
        files = oracle_files(video)
        parsed = [parse_query(path) for path in files]
        query_texts = [f"视频中同时出现{concept_zh[a]}和{concept_zh[b]}的相关画面" for a, b in parsed]
        unique_concepts = sorted({concept for pair in parsed for concept in pair})
        english_names = {
            "Cupglass": "cup or glass", "Musicalinstrument": "musical instrument",
            "Petsanimal": "pet or animal",
        }
        prompts = [f"a video frame showing {english_names.get(name, name.lower())}" for name in unique_concepts]
        with torch.no_grad():
            tokens = clip.tokenize(prompts).to(device)
            encoded = english_clip.encode_text(tokens)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            concept_features = encoded.cpu().numpy().astype(np.float32)
        concept_to_feature = dict(zip(unique_concepts, concept_features))
        for query_index, (path, (concept1, concept2), query_text) in enumerate(zip(files, parsed, query_texts)):
            # IntentParser represents query targets as a focus list. Score each
            # focus independently and fuse them symmetrically, matching that
            # structured representation and the original QFVS two-concept task.
            semantic1 = image_features @ concept_to_feature[concept1]
            semantic2 = image_features @ concept_to_feature[concept2]
            zero_shot_semantic = minmax(0.5 * semantic1 + 0.5 * semantic2)
            concept1_index = list(concept_zh).index(concept1)
            concept2_index = list(concept_zh).index(concept2)
            calibrated_semantic = minmax(
                0.5 * calibrated_concepts[:, concept1_index]
                + 0.5 * calibrated_concepts[:, concept2_index]
            )
            semantic = zero_shot_semantic
            adaptive, diagnostics = confidence_adaptive_fusion(
                visual, semantic, semantic_prior=0.8, prior_strength=0.5
            )
            adapter_adaptive, adapter_diagnostics = confidence_adaptive_fusion(
                visual, calibrated_semantic, semantic_prior=0.9, prior_strength=0.5
            )
            n_shots = len(visual)
            k = max(1, int(n_shots * 0.02))
            variants = {
                "Uniform": np.linspace(0, n_shots - 1, k, dtype=int),
                "Visual-BiLSTM": np.argsort(-visual, kind="stable")[:k],
                "CLIP-zero-shot": np.argsort(-zero_shot_semantic, kind="stable")[:k],
                "Concept-Adapter": np.argsort(-calibrated_semantic, kind="stable")[:k],
                "Concept-Adapter-MMR": mmr_select(calibrated_semantic, image_features, k, diversity=0.15),
                "Adapter-Adaptive-no-Diversity": np.argsort(-adapter_adaptive, kind="stable")[:k],
                "Ours-Adapter-Fusion": mmr_select(adapter_adaptive, image_features, k, diversity=0.15),
                "Fixed-Fusion": np.argsort(-(0.2 * visual + 0.8 * semantic), kind="stable")[:k],
                "Adaptive-no-Diversity": np.argsort(-adaptive, kind="stable")[:k],
                "ZeroShot-Fusion-MMR": mmr_select(adaptive, image_features, k, diversity=0.15),
            }
            oracle = [int(line.strip()) - 1 for line in path.read_text("utf-8").splitlines() if line.strip()]
            for method, predicted in variants.items():
                metrics = evaluate_semantic_summary(predicted, oracle, tags[video_index])
                auxiliary = evaluate_exact_and_query_relevance(
                    predicted, oracle, tags[video_index], concept1_index, concept2_index
                )
                rows.append({
                    "video": video, "query_id": path.stem.removesuffix("_oracle"),
                    "concept1": concept1, "concept2": concept2, "query_zh": query_text,
                    "method": method, "n_shots": n_shots, "selected_shots": len(predicted),
                    "oracle_shots": len(oracle), "precision": metrics["precision"],
                    "recall": metrics["recall"], "fscore": metrics["fscore"],
                    **auxiliary,
                    "selected_indices": json.dumps([int(value) for value in predicted]),
                    "adaptive_semantic_weight": diagnostics["semantic_weight"],
                })
        print(video, len(files), "queries", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(OUT, len(rows))


if __name__ == "__main__":
    main()
