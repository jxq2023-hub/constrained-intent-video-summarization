"""Run full-video TVSum/SumMe experiments under the standard 15% protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from research_q3.features.sampled_video_scorer import SampledVideoScorer
from research_q3.methods.adaptive_fusion import confidence_adaptive_fusion
from research_q3.protocols.generic_summary import evaluate_fscore, expand_sample_scores, generate_summary


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research_q3" / "data" / "standard_protocol"
CACHE = ROOT / "research_q3" / "data" / "feature_cache"
RESULTS = ROOT / "research_q3" / "results"
MANIFEST = ROOT / "research_q3" / "configs" / "query_manifest.json"


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    span = float(values.max() - values.min())
    return (values - values.min()) / span if span > 1e-12 else np.zeros_like(values)


def tvsum_paths() -> dict[str, Path]:
    return {path.stem: path for path in (ROOT / "training" / "data" / "tvsum" / "video").glob("*.mp4")}


def summe_paths() -> dict[str, Path]:
    return {path.stem: path for path in (ROOT / "training" / "data" / "summe" / "video").glob("*.mp4")}


def dataset_config(name: str) -> tuple[Path, str, dict[str, Path]]:
    if name == "TVSum":
        return DATA / "eccv16_dataset_tvsum_google_pool5.h5", "avg", tvsum_paths()
    return DATA / "eccv16_dataset_summe_google_pool5.h5", "max", summe_paths()


def resolve_video(name: str, key: str, group: h5py.Group, paths: dict[str, Path], manifest: dict) -> tuple[Path, str]:
    if name == "TVSum":
        ordered_ids = list(manifest["TVSum"])
        video_id = ordered_ids[int(key.split("_")[-1]) - 1]
        return paths[video_id], manifest["TVSum"][video_id]["query_zh"]
    raw = group["video_name"][()]
    title = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    return paths[title], manifest["SumMe"][key]["query_zh"]


def score_variants(visual: np.ndarray, semantic: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    visual_n, semantic_n = minmax(visual), minmax(semantic)
    adaptive, diagnostics = confidence_adaptive_fusion(
        visual_n, semantic_n, semantic_prior=0.65, prior_strength=0.5
    )
    return {
        "Visual-BiLSTM": visual_n,
        "CLIP-only": semantic_n,
        "Fixed-Fusion": 0.35 * visual_n + 0.65 * semantic_n,
        "Ours-Adaptive": adaptive,
    }, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["TVSum", "SumMe", "all"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text("utf-8"))
    datasets = ["TVSum", "SumMe"] if args.dataset == "all" else [args.dataset]
    scorer = SampledVideoScorer(device=args.device, batch_size=args.batch_size)
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    diagnostics_rows: list[dict] = []

    for dataset in datasets:
        h5_path, aggregation, videos = dataset_config(dataset)
        with h5py.File(h5_path, "r") as handle:
            keys = sorted(handle, key=lambda item: int(item.split("_")[-1]))
            if args.limit:
                keys = keys[: args.limit]
            for position, key in enumerate(keys, 1):
                group = handle[key]
                video_path, query = resolve_video(dataset, key, group, videos, manifest)
                cache_path = CACHE / f"{dataset}_{key}.npz"
                if cache_path.exists():
                    cache = np.load(cache_path)
                    visual, semantic = cache["visual_scores"], cache["semantic_scores"]
                else:
                    scored = scorer.score_video(video_path, group["picks"][:], query)
                    visual, semantic = scored["visual_scores"], scored["semantic_scores"]
                    np.savez_compressed(
                        cache_path, indices=scored["indices"], visual_scores=visual,
                        semantic_scores=semantic, query=np.asarray(query), video=np.asarray(str(video_path)),
                    )
                variants, diagnostics = score_variants(visual, semantic)
                n_frames = int(group["n_frames"][()])
                for method, sampled_scores in variants.items():
                    dense = expand_sample_scores(sampled_scores, group["picks"][:], n_frames)
                    summary = generate_summary(dense, group["change_points"][:], n_frames)
                    metrics = evaluate_fscore(summary, group["user_summary"][:], aggregation)
                    rows.append({
                        "dataset": dataset, "video_key": key, "method": method,
                        "query": query, "fscore": metrics["fscore"],
                        "precision_mean": metrics["precision"], "recall_mean": metrics["recall"],
                        "selected_ratio": float(summary.mean()),
                    })
                diagnostics_rows.append({"dataset": dataset, "video_key": key, **diagnostics})
                print(f"[{dataset} {position}/{len(keys)}] {key} cached={cache_path.exists()}", flush=True)

    for filename, data in (("standard_protocol_results.csv", rows), ("adaptive_fusion_diagnostics.csv", diagnostics_rows)):
        path = RESULTS / filename
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
        print(path)


if __name__ == "__main__":
    main()
