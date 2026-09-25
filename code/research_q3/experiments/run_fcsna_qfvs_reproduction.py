"""Protocol-controlled reproduction of public FCSNA-QFVS (ICARCV 2024).

Source architecture and hyperparameters:
https://github.com/srkds/FCSNA-QFVS

Controls relative to the public Colab notebook:
* the held-out video is evaluated once after a fixed 20 epochs, never used for
  checkpoint selection;
* deterministic seeds and stable file ordering are used;
* selected shot indices are evaluated by this project's common 2% QFVS
  semantic-matching protocol and auxiliary tag-grounded metrics;
* the paper-defined arithmetic mean of two concept vectors, the intended
  shot-feature axis order, and the global-attention mask order are explicit
  (the public notebook contains multiplication/reshape ambiguities).

This is a local reproduction, not a quotation of the paper's reported score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from research_q3.protocols.qfvs_semantic import (
    evaluate_exact_and_query_relevance,
    evaluate_semantic_summary,
    load_qfvs_tags,
)


ROOT = Path(__file__).resolve().parents[2]
QFVS = ROOT / "research_q3" / "data" / "qfvs"
ANNOTATIONS = QFVS / "annotations"
PUBLIC_DATA = ROOT / "research_q3" / "third_party" / "chan_qfvs" / "CHAN-QFVS-PyTorch-Implementation-main" / "data"
FEATURES = PUBLIC_DATA / "processed"
EMBEDDINGS = PUBLIC_DATA / "query_dictionary.pkl"
RESULTS = ROOT / "research_q3" / "results"
RAW_OUT = RESULTS / "fcsna_qfvs_reproduction_results.csv"
SUMMARY_OUT = RESULTS / "fcsna_qfvs_reproduction_summary.csv"
CONFIG_OUT = RESULTS / "fcsna_qfvs_reproduction_config.json"
REPORT_OUT = RESULTS / "fcsna_qfvs_reproduction_report.md"
CHECKPOINTS = RESULTS / "fcsna_checkpoints"
SEED = 20260720
TRANSFER = {"Cupglass": "Glass", "Musicalinstrument": "Instrument", "Petsanimal": "Animal"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_embeddings() -> dict[str, np.ndarray]:
    with EMBEDDINGS.open("rb") as stream:
        return pickle.load(stream)


def query_files(video_id: int) -> list[Path]:
    return sorted((ANNOTATIONS / "Oracle_summaries" / f"P0{video_id}").glob("*_oracle.txt"))


def query_pair(path: Path) -> tuple[str, str]:
    fields = path.stem.removesuffix("_oracle").split("_")
    return fields[0], fields[1]


class FCSNADataset(Dataset):
    def __init__(self, videos: list[int]):
        self.examples = [(video, path) for video in videos for path in query_files(video)]
        self.embedding = load_embeddings()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        video_id, path = self.examples[index]
        with h5py.File(FEATURES / f"V{video_id}_resnet_avg.h5", "r") as source:
            features = torch.tensor(source["features"][()], dtype=torch.float32)
            seg_len = torch.tensor(source["seg_len"][()], dtype=torch.long)
        concept1, concept2 = query_pair(path)
        dense_path = ANNOTATIONS / "Dense_per_shot_tags" / f"P0{video_id}" / f"P0{video_id}.txt"
        labels1, labels2 = torch.zeros(4000), torch.zeros(4000)
        for shot, line in enumerate(dense_path.read_text("utf-8").splitlines()):
            tags = line.strip().split(",")
            labels1[shot] = float(concept1 in tags)
            labels2[shot] = float(concept2 in tags)
        shots = int(seg_len.sum())
        valid_labels = torch.arange(4000) < shots
        key1, key2 = TRANSFER.get(concept1, concept1), TRANSFER.get(concept2, concept2)
        embedding1 = torch.tensor(self.embedding[key1], dtype=torch.float32)
        embedding2 = torch.tensor(self.embedding[key2], dtype=torch.float32)
        return features, seg_len, embedding1, embedding2, labels1, labels2, valid_labels


class Attention(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, head_size: int):
        super().__init__()
        self.query = nn.Linear(query_dim, head_size, bias=False)
        self.key = nn.Linear(key_dim, head_size, bias=False)
        self.value = nn.Linear(key_dim, head_size, bias=False)

    def forward(self, query_input, key_input, attention_mask):
        query = self.query(query_input)
        key = self.key(key_input)
        value = self.value(key_input)
        weights = query @ key.transpose(1, 2) * (1.0 / math.sqrt(query.size(-1)))
        weights = weights.masked_fill(~attention_mask, torch.finfo(weights.dtype).min)
        weights = F.softmax(weights, dim=-1)
        return weights @ value


class FCSNA(nn.Module):
    """Public FCSNA fully convolutional sequence architecture."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ("conv1_1", nn.Conv1d(2048, 2048, 3, padding=1)), ("bn1_1", nn.BatchNorm1d(2048)),
            ("relu1_1", nn.ReLU(inplace=True)), ("conv1_2", nn.Conv1d(2048, 2048, 3, padding=1)),
            ("bn1_2", nn.BatchNorm1d(2048)), ("relu1_2", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool1d(2, stride=2, ceil_mode=True)),
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ("conv2_1", nn.Conv1d(2048, 2048, 3, padding=1)), ("bn2_1", nn.BatchNorm1d(2048)),
            ("relu2_1", nn.ReLU(inplace=True)), ("conv2_2", nn.Conv1d(2048, 2048, 3, padding=1)),
            ("bn2_2", nn.BatchNorm1d(2048)), ("relu2_2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool1d(2, stride=2, ceil_mode=True)),
        ]))
        self.conv3 = nn.Sequential(OrderedDict([
            ("conv3_1", nn.Conv1d(2048, 2048, 3, padding=1)), ("bn3_1", nn.BatchNorm1d(2048)),
            ("relu3_1", nn.ReLU(inplace=True)), ("conv3_2", nn.Conv1d(2048, 2048, 3, padding=1)),
            ("bn3_2", nn.BatchNorm1d(2048)), ("relu3_2", nn.ReLU(inplace=True)),
            ("conv3_3", nn.Conv1d(2048, 2048, 3, padding=1)), ("bn3_3", nn.BatchNorm1d(2048)),
            ("relu3_3", nn.ReLU(inplace=True)), ("pool3", nn.MaxPool1d(2, stride=2, ceil_mode=True)),
        ]))
        self.conv4 = nn.Sequential(OrderedDict([
            ("conv4_1", nn.Conv1d(2048, 2048, 3)), ("bn4_1", nn.BatchNorm1d(2048)),
            ("relu4_1", nn.ReLU(inplace=True)), ("conv4_2", nn.Conv1d(2048, 2048, 3)),
            ("bn4_2", nn.BatchNorm1d(2048)), ("relu4_2", nn.ReLU(inplace=True)),
            ("conv4_3", nn.Conv1d(2048, 2048, 3)), ("bn4_3", nn.BatchNorm1d(2048)),
            ("relu4_3", nn.ReLU(inplace=True)),
        ]))
        self.conv5 = nn.Sequential(OrderedDict([
            ("conv5_1", nn.Conv1d(2048, 2048, 3, padding=1)), ("bn5_1", nn.BatchNorm1d(2048)),
            ("relu5_1", nn.ReLU(inplace=True)), ("conv5_2", nn.Conv1d(2048, 2048, 3, padding=1)),
            ("bn5_2", nn.BatchNorm1d(2048)), ("relu5_2", nn.ReLU(inplace=True)),
            ("conv5_3", nn.Conv1d(2048, 2048, 3, padding=1)), ("bn5_3", nn.BatchNorm1d(2048)),
            ("relu5_3", nn.ReLU(inplace=True)), ("pool5", nn.MaxPool1d(2, stride=2, ceil_mode=True)),
        ]))
        self.conv6 = nn.Sequential(nn.Conv1d(2048, 4096, 1), nn.BatchNorm1d(4096), nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.conv7 = nn.Sequential(nn.Conv1d(4096, 4096, 1), nn.BatchNorm1d(4096), nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.conv8 = nn.Sequential(nn.Conv1d(4096, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True))
        self.conv_pool4 = nn.Conv1d(2048, 1024, 1)
        self.bn_pool4 = nn.BatchNorm1d(1024)
        self.self_attention = Attention(256, 256, 256)
        self.query_relevance_attention = Attention(300, 256, 256)
        self.global_attention = Attention(256, 256, 256)
        self.deconv1 = nn.ConvTranspose1d(768, 1024, 3, padding=1, stride=2, bias=False)
        self.deconv2 = nn.ConvTranspose1d(1024, 1024, 20, stride=10, bias=False)
        self.similarity_linear1 = nn.Linear(1024, 300, bias=False)
        self.similarity_linear2 = nn.Linear(300, 300, bias=False)
        self.mlp = nn.Linear(1200, 1)

    def forward(self, batch, seg_len, concept1, concept2):
        batch_size, segments, length, _ = batch.shape
        hidden = batch.reshape(batch_size * segments, length, -1).transpose(1, 2)
        hidden = self.conv1(hidden)
        hidden = self.conv2(hidden)
        hidden = self.conv3(hidden)
        hidden = self.conv4(hidden)
        pool4 = hidden
        hidden = self.conv5(hidden)
        hidden = self.conv6(hidden)
        hidden = self.conv7(hidden)
        hidden = self.conv8(hidden).transpose(1, 2)
        reduced_length = hidden.shape[1]

        steps = torch.arange(reduced_length, device=batch.device).view(1, 1, -1)
        valid_steps = steps < torch.ceil(seg_len.to(batch.device).unsqueeze(-1) / 20.0).long()
        local_mask = valid_steps.reshape(batch_size * segments, 1, reduced_length)
        self_result = self.self_attention(hidden, hidden, local_mask)

        # The paper defines Hq as the average of both concept features.  The
        # notebook uses elementwise multiplication here; we follow the paper.
        average_query = (concept1 + concept2) / 2.0
        tiled_query = average_query[:, None, None, :].expand(
            batch_size, segments, reduced_length, 300
        ).reshape(batch_size * segments, reduced_length, 300)
        relevance = self.query_relevance_attention(tiled_query, hidden, local_mask)
        relevance = relevance.mean(dim=1)[None, :, :].expand(reduced_length, -1, -1)

        time_major = hidden.reshape(batch_size, segments, reduced_length, 256).permute(2, 0, 1, 3).reshape(
            reduced_length, batch_size * segments, 256
        )
        global_mask = valid_steps.permute(2, 0, 1).reshape(reduced_length, 1, batch_size * segments)
        global_result = self.global_attention(time_major, relevance, global_mask)
        global_result = global_result.reshape(reduced_length, batch_size, segments, 256).permute(1, 2, 0, 3)
        global_result = global_result.reshape(batch_size * segments, reduced_length, 256)

        fused = torch.cat((hidden, self_result, global_result), dim=-1).transpose(1, 2)
        upscore = self.deconv1(fused)
        skip = self.bn_pool4(self.conv_pool4(pool4))
        decoded = self.deconv2(upscore + skip).transpose(1, 2)

        # decoded is (B*segment, shot, channel); preserve that shot axis when
        # flattening segments.  The notebook transposes it once more before
        # reshape, which interleaves channel and temporal positions.
        shot_features = decoded.contiguous().reshape(batch_size, segments * length, -1)
        shot_features = self.similarity_linear1(shot_features)
        query1 = self.similarity_linear2(concept1)
        query2 = self.similarity_linear2(concept2)

        def score(query):
            expanded = query[:, None, :].expand(-1, segments * length, -1)
            added = expanded + shot_features
            multiplied = expanded * shot_features
            return torch.sigmoid(self.mlp(torch.cat((expanded, shot_features, added, multiplied), dim=-1)))

        return score(query1).squeeze(-1).reshape(batch_size, segments, length), score(query2).squeeze(-1).reshape(batch_size, segments, length)


def feature_mask(seg_len: torch.Tensor, device: torch.device) -> torch.Tensor:
    positions = torch.arange(200, device=device).view(1, 1, -1)
    return positions < seg_len.to(device).unsqueeze(-1)


def train_fold(test_video: int, args, device: torch.device) -> FCSNA:
    fold_seed = args.seed + test_video
    seed_everything(fold_seed)
    train_videos = [video for video in [1, 2, 3, 4] if video != test_video]
    generator = torch.Generator().manual_seed(fold_seed)
    loader = DataLoader(
        FCSNADataset(train_videos), batch_size=args.batch_size, shuffle=True,
        num_workers=0, generator=generator,
    )
    model = FCSNA().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for features, seg_len, query1, query2, labels1, labels2, valid_labels in loader:
            features, seg_len = features.to(device), seg_len.to(device)
            query1, query2 = query1.to(device), query2.to(device)
            labels1, labels2, valid_labels = labels1.to(device), labels2.to(device), valid_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                scores1, scores2 = model(features, seg_len, query1, query2)
                mask = feature_mask(seg_len, device)
                loss = torch.zeros((), device=device)
                for sample in range(len(features)):
                    loss = loss + criterion(scores1[sample][mask[sample]], labels1[sample][valid_labels[sample]])
                    loss = loss + criterion(scores2[sample][mask[sample]], labels2[sample][valid_labels[sample]])
                loss = loss / len(features)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        print(f"fold={test_video} epoch={epoch + 1}/{args.epochs} loss={np.mean(losses):.6f}", flush=True)
    return model


@torch.no_grad()
def evaluate_fold(model: FCSNA, test_video: int, device: torch.device) -> list[dict]:
    model.eval()
    embeddings = load_embeddings()
    tags = load_qfvs_tags(str(ANNOTATIONS / "Tags.mat"))
    with h5py.File(FEATURES / f"V{test_video}_resnet_avg.h5", "r") as source:
        features = torch.tensor(source["features"][()], dtype=torch.float32, device=device).unsqueeze(0)
        seg_len = torch.tensor(source["seg_len"][()], dtype=torch.long, device=device).unsqueeze(0)
    mask = feature_mask(seg_len, device)[0]
    rows = []
    for path in query_files(test_video):
        concept1, concept2 = query_pair(path)
        query1 = torch.tensor(embeddings[TRANSFER.get(concept1, concept1)], dtype=torch.float32, device=device).unsqueeze(0)
        query2 = torch.tensor(embeddings[TRANSFER.get(concept2, concept2)], dtype=torch.float32, device=device).unsqueeze(0)
        scores1, scores2 = model(features, seg_len, query1, query2)
        scores = (scores1[0] + scores2[0])[mask]
        k = max(1, int(len(scores) * 0.02))
        selected = torch.topk(scores, k=k).indices.detach().cpu().numpy().astype(int).tolist()
        oracle = [int(line) - 1 for line in path.read_text("utf-8").splitlines() if line.strip()]
        video_tags = tags[test_video - 1]
        concept_names = list(json.loads((ROOT / "research_q3" / "configs" / "qfvs_concepts.json").read_text("utf-8")))
        official = evaluate_semantic_summary(selected, oracle, video_tags)
        auxiliary = evaluate_exact_and_query_relevance(
            selected, oracle, video_tags, concept_names.index(concept1), concept_names.index(concept2)
        )
        rows.append({
            "video": f"P0{test_video}", "query_id": path.stem.removesuffix("_oracle"),
            "concept1": concept1, "concept2": concept2,
            "method": "FCSNA-QFVS-Controlled-Reproduction", "selected_shots": len(selected),
            "selected_indices": json.dumps(selected), **official, **auxiliary,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=["1", "2", "3", "4", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="use CUDA mixed precision if memory constrained")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    folds = [1, 2, 3, 4] if args.fold == "all" else [int(args.fold)]
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    previous = []
    if RAW_OUT.exists() and not args.fresh:
        import pandas as pd
        previous = pd.read_csv(RAW_OUT).to_dict("records")
    completed = {int(str(row["video"])[-1]) for row in previous}
    rows = [row for row in previous if int(str(row["video"])[-1]) not in folds or int(str(row["video"])[-1]) in completed]
    for fold in folds:
        if fold in completed and not args.fresh:
            print(f"fold={fold} already cached", flush=True)
            continue
        model = train_fold(fold, args, device)
        fold_rows = evaluate_fold(model, fold, device)
        rows.extend(fold_rows)
        torch.save({"model": model.state_dict(), "args": vars(args), "fold": fold}, CHECKPOINTS / f"fold{fold}.pt")
        with RAW_OUT.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    import pandas as pd
    frame = pd.DataFrame(rows)
    metrics = ["precision", "recall", "fscore", "exact_fscore", "query_relevance_or", "query_relevance_and"]
    summary = frame.groupby("method", as_index=False).agg(n=("query_id", "size"), **{
        metric: (metric, "mean") for metric in metrics
    })
    per_video = frame.groupby("video", as_index=False)[metrics].mean()
    summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")
    config = {
        **vars(args), "source_url": "https://github.com/srkds/FCSNA-QFVS",
        "paper_doi": "10.1109/ICARCV63323.2024.10821560",
        "test_selection": "fixed final epoch; held-out video never used for checkpoint selection",
        "budget": "top 2% shots", "completed_folds": sorted(frame.video.unique().tolist()),
    }
    CONFIG_OUT.write_text(json.dumps(config, ensure_ascii=False, indent=2), "utf-8")
    report = [
        "# FCSNA-QFVS协议受控复现", "", summary.to_markdown(index=False, floatfmt=".4f"), "",
        "## 分视频本地结果", "", per_video.to_markdown(index=False, floatfmt=".4f"), "",
        "原论文报告四个视频F1分别为0.4515、0.5032、0.5724和0.3720，平均0.4747；",
        "本地统一协议值与其不可直接互换。原论文描述每视频46个查询，而当前公开Oracle文件",
        "提供45个查询，本文与其他方法统一使用共同的180个查询—视频单元。", "",
        "该结果为公开架构在本地统一协议下的复现值，不是原论文表格的直接引用。",
        "与公开Notebook不同，测试视频不参与epoch选择；固定训练20个epoch并报告最终模型。",
        "按论文定义，两概念向量采用算术平均；公共Notebook中的逐元素乘法、输出时间轴重排和",
        "全局注意力掩码轴顺序歧义均按论文张量语义修正，其余网络与超参数保持一致。",
    ]
    REPORT_OUT.write_text("\n".join(report), "utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
