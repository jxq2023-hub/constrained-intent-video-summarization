"""Train CHAN in four leave-one-video-out folds and evaluate agent fusion.

The CHAN architecture and precomputed ResNet/GloVe inputs come from the public
implementation.  Every fold is trained for a fixed five epochs; the held-out
video is never used for fitting or epoch selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from research_q3.methods.adaptive_fusion import confidence_adaptive_fusion
from research_q3.methods.diverse_selection import mmr_select
from research_q3.protocols.qfvs_semantic import (
    evaluate_exact_and_query_relevance, evaluate_semantic_summary, load_qfvs_tags,
)


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "research_q3" / "third_party" / "chan_qfvs" / "CHAN-QFVS-PyTorch-Implementation-main" / "data"
CHAN_ROOT = ROOT / "research_q3" / "third_party" / "chan"
QFVS = ROOT / "research_q3" / "data" / "qfvs"
OUT = ROOT / "research_q3" / "results" / "qfvs_chan_fusion_results.csv"
CHECKPOINTS = ROOT / "research_q3" / "results" / "checkpoints"
sys.path.insert(0, str(CHAN_ROOT))
from model.CHAN import CHAN  # noqa: E402


CONFIG = {
    "max_segment_num": 20, "max_frame_num": 200, "similarity_dim": 1000,
    "concept_dim": 300, "in_channel": 2048, "conv1_channel": 512,
    "conv2_channel": 256, "deconv1_channel": 1024, "deconv2_channel": 1024,
}
TRANSFER = {"Cupglass": "Glass", "Musicalinstrument": "Instrument", "Petsanimal": "Animal"}


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_inputs():
    features, seg_lens = {}, {}
    for video_id in range(1, 5):
        with h5py.File(PUBLIC / "processed" / f"V{video_id}_resnet_avg.h5", "r") as handle:
            features[video_id] = torch.from_numpy(handle["features"][:]).float()
            seg_lens[video_id] = torch.from_numpy(handle["seg_len"][:]).long()
    with (PUBLIC / "query_dictionary.pkl").open("rb") as stream:
        embeddings = pickle.load(stream)
    return features, seg_lens, embeddings


class CHANDataset(Dataset):
    def __init__(self, video_ids, features, seg_lens, embeddings):
        self.features, self.seg_lens, self.embeddings = features, seg_lens, embeddings
        self.items = []
        for video_id in video_ids:
            dense = (QFVS / "annotations" / "Dense_per_shot_tags" / f"P0{video_id}" / f"P0{video_id}.txt").read_text().splitlines()
            for path in sorted((QFVS / "annotations" / "Oracle_summaries" / f"P0{video_id}").glob("*_oracle.txt")):
                concept1, concept2 = path.stem.removesuffix("_oracle").split("_")
                y1 = torch.zeros(4000); y2 = torch.zeros(4000)
                for index, line in enumerate(dense):
                    labels = line.split(",")
                    y1[index] = float(concept1 in labels); y2[index] = float(concept2 in labels)
                mask = torch.zeros(4000, dtype=torch.bool); mask[:len(dense)] = True
                self.items.append((video_id, concept1, concept2, y1, y2, mask))

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        video_id, c1, c2, y1, y2, mask = self.items[index]
        q1 = torch.from_numpy(np.asarray(self.embeddings[TRANSFER.get(c1, c1)], dtype=np.float32))
        q2 = torch.from_numpy(np.asarray(self.embeddings[TRANSFER.get(c2, c2)], dtype=np.float32))
        return self.features[video_id], self.seg_lens[video_id], q1, q2, y1, y2, mask


def valid_mask(seg_len: torch.Tensor, device: str) -> torch.Tensor:
    batch = len(seg_len)
    mask = torch.zeros(batch, 20, 200, dtype=torch.bool, device=device)
    for i in range(batch):
        for j, length in enumerate(seg_len[i]):
            mask[i, j, : int(length)] = True
    return mask


def train_fold(test_video: int, features, seg_lens, embeddings, seed: int = 20260717) -> CHAN:
    seed_all(seed + test_video)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CHAN(CONFIG).to(device)
    dataset = CHANDataset([v for v in range(1, 5) if v != test_video], features, seg_lens, embeddings)
    loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    epochs = 20
    for epoch in range(epochs):
        model.train(); losses = []
        for x, lengths, q1, q2, y1, y2, gt_mask in loader:
            x, lengths, q1, q2 = x.to(device), lengths.to(device), q1.to(device), q2.to(device)
            gt_mask, y1, y2 = gt_mask.to(device), y1.to(device), y2.to(device)
            p1, p2 = model(x, lengths, q1, q2)
            loss = torch.zeros((), device=device)
            for i in range(len(x)):
                n_valid = int(gt_mask[i].sum())
                loss = loss + F.binary_cross_entropy(p1[i].reshape(-1)[:n_valid], y1[i, :n_valid])
                loss = loss + F.binary_cross_entropy(p2[i].reshape(-1)[:n_valid], y2[i, :n_valid])
            loss = loss / len(x)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        print(f"fold={test_video} epoch={epoch+1}/{epochs} loss={np.mean(losses):.6f}", flush=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINTS / f"chan_fold_{test_video}_seed_{seed}.pth")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features, seg_lens, embeddings = load_inputs()
    tags = load_qfvs_tags(str(QFVS / "annotations" / "Tags.mat"))
    concept_names = list(json.loads((ROOT / "research_q3/configs/qfvs_concepts.json").read_text("utf-8")))
    rows = []
    for test_video in range(1, 5):
        if args.eval_only:
            model = CHAN(CONFIG).to(device)
            model.load_state_dict(torch.load(CHECKPOINTS / f"chan_fold_{test_video}_seed_20260717.pth", map_location=device))
        else:
            model = train_fold(test_video, features, seg_lens, embeddings).to(device)
        model.eval()
        video = f"P0{test_video}"; n_shots = len(tags[test_video - 1]); k = max(1, int(n_shots * 0.02))
        visual = np.load(QFVS / "feature_cache" / f"{video}.npz")["visual_scores"][:n_shots]
        diversity_features = np.load(QFVS / "feature_cache" / f"{video}.npz")["clip_image_features"][:n_shots].astype(np.float32)
        x = features[test_video].unsqueeze(0).to(device); lengths = seg_lens[test_video].unsqueeze(0).to(device)
        for path in sorted((QFVS / "annotations" / "Oracle_summaries" / video).glob("*_oracle.txt")):
            c1, c2 = path.stem.removesuffix("_oracle").split("_")
            q1 = torch.from_numpy(np.asarray(embeddings[TRANSFER.get(c1, c1)], dtype=np.float32)).unsqueeze(0).to(device)
            q2 = torch.from_numpy(np.asarray(embeddings[TRANSFER.get(c2, c2)], dtype=np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                p1, p2 = model(x, lengths, q1, q2)
                chan_score = (0.5 * p1.reshape(-1)[:n_shots] + 0.5 * p2.reshape(-1)[:n_shots]).cpu().numpy()
            adaptive, diagnostics = confidence_adaptive_fusion(visual, chan_score, semantic_prior=0.85, prior_strength=0.5)
            variants = {
                "CHAN": np.argsort(-chan_score, kind="stable")[:k],
                "CHAN-MMR": mmr_select(chan_score, diversity_features, k, diversity=0.15),
                "Ours-CHAN-Fusion": mmr_select(adaptive, diversity_features, k, diversity=0.15),
            }
            oracle = [int(value) - 1 for value in path.read_text().split()]
            c1_index, c2_index = concept_names.index(c1), concept_names.index(c2)
            for method, predicted in variants.items():
                result = evaluate_semantic_summary(predicted, oracle, tags[test_video - 1])
                auxiliary = evaluate_exact_and_query_relevance(
                    predicted, oracle, tags[test_video - 1], c1_index, c2_index
                )
                rows.append({"video": video, "query_id": path.stem.removesuffix("_oracle"), "method": method,
                             "precision": result["precision"], "recall": result["recall"], "fscore": result["fscore"],
                             **auxiliary, "selected_indices": json.dumps([int(value) for value in predicted]),
                             "selected_shots": len(predicted), "semantic_weight": diagnostics["semantic_weight"]})
        print(f"evaluated {video}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(OUT, len(rows))


if __name__ == "__main__":
    main()
