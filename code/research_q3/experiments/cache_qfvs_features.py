"""Extract frozen Visual-BiLSTM scores and CLIP image features for QFVS."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from research_q3.features.sampled_video_scorer import SampledVideoScorer


ROOT = Path(__file__).resolve().parents[2]
FRAMES = ROOT / "research_q3" / "data" / "qfvs" / "shot_frames"
CACHE = ROOT / "research_q3" / "data" / "qfvs" / "feature_cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", choices=["P01", "P02", "P03", "P04", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device")
    args = parser.parse_args()
    videos = ["P01", "P02", "P03", "P04"] if args.video == "all" else [args.video]
    scorer = SampledVideoScorer(device=args.device, batch_size=args.batch_size)
    CACHE.mkdir(parents=True, exist_ok=True)
    for video in videos:
        output = CACHE / f"{video}.npz"
        paths = sorted((FRAMES / video).glob("*.jpg"))
        features = scorer.score_image_paths(paths)
        np.savez_compressed(
            output,
            visual_scores=features["visual_scores"],
            clip_image_features=features["clip_image_features"].astype(np.float16),
            frame_names=np.asarray([path.name for path in paths]),
        )
        print(video, len(paths), output, flush=True)


if __name__ == "__main__":
    main()
