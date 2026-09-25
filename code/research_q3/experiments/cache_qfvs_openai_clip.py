"""Cache official OpenAI CLIP ViT-B/32 shot embeddings for English QFVS."""

from __future__ import annotations

import argparse
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FRAMES = ROOT / "research_q3" / "data" / "qfvs" / "shot_frames"
CACHE = ROOT / "research_q3" / "data" / "qfvs" / "feature_cache"
WEIGHTS = ROOT / "models" / "ViT-B-32.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(str(WEIGHTS), device=device, jit=False)
    model.eval()
    for video in ["P01", "P02", "P03", "P04"]:
        paths = sorted((FRAMES / video).glob("*.jpg"))
        chunks = []
        with torch.no_grad():
            for start in range(0, len(paths), args.batch_size):
                images = torch.stack([
                    preprocess(Image.open(path).convert("RGB"))
                    for path in paths[start : start + args.batch_size]
                ]).to(device)
                values = model.encode_image(images)
                values = values / values.norm(dim=-1, keepdim=True)
                chunks.append(values.cpu().numpy().astype(np.float16))
        output = CACHE / f"{video}_openai_clip.npz"
        np.savez_compressed(output, image_features=np.concatenate(chunks), frame_names=np.asarray([p.name for p in paths]))
        print(video, len(paths), output, flush=True)


if __name__ == "__main__":
    main()
