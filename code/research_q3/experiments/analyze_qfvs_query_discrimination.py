"""Query-swap negative control for personalized QFVS summaries.

For every generated summary, compare its tag-grounded relevance to the true
query against all other 44 queries from the same video.  The discrimination
AUC is the probability that the true query receives a higher score than a
random wrong query (ties count as 0.5). Query-independent summaries should
average approximately 0.5 across the balanced set of queries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from research_q3.protocols.qfvs_semantic import load_qfvs_tags


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research_q3" / "results"
BASE = RESULTS / "qfvs_results.csv"
NESTED = RESULTS / "qfvs_nested_selection_results.csv"
CHAN = RESULTS / "qfvs_chan_fusion_results.csv"
CONCEPTS = ROOT / "research_q3" / "configs" / "qfvs_concepts.json"
TAGS = ROOT / "research_q3" / "data" / "qfvs" / "annotations" / "Tags.mat"
OUT = RESULTS / "qfvs_query_discrimination_results.csv"
SUMMARY = RESULTS / "qfvs_query_discrimination_summary.csv"
PAIRWISE = RESULTS / "qfvs_query_discrimination_pairwise.csv"
REPORT = RESULTS / "qfvs_query_discrimination_report.md"
METHODS = [
    "Uniform", "Visual-BiLSTM", "CLIP-zero-shot", "Concept-Adapter-MMR",
    "CHAN-MMR", "Ours-Nested-Intent-Selector",
]
SEED = 20260719


def parse_indices(value: str) -> list[int]:
    return [int(item) for item in json.loads(value)]


def relevance(selected: list[int], labels: np.ndarray, c1: int, c2: int) -> tuple[float, float]:
    chosen = labels[selected]
    first = chosen[:, c1] > 0
    second = chosen[:, c2] > 0
    return float(np.mean(first | second)), float(np.mean(first & second))


def auc_against_wrong(true_value: float, wrong_values: np.ndarray) -> float:
    return float(np.mean((true_value > wrong_values) + 0.5 * (true_value == wrong_values)))


def cluster_bootstrap(frame: pd.DataFrame, column: str, iterations: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    videos = frame.video.unique()
    video_values = frame.groupby("video")[column].mean()
    draws = rng.choice(videos, size=(iterations, len(videos)), replace=True)
    means = np.array([[video_values.loc[item] for item in row] for row in draws]).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def main() -> None:
    names = list(json.loads(CONCEPTS.read_text("utf-8")))
    concept_index = {name: index for index, name in enumerate(names)}
    tags = load_qfvs_tags(str(TAGS))
    base = pd.read_csv(BASE)
    meta = base[["video", "query_id", "concept1", "concept2"]].drop_duplicates()
    selected = base[base.method.isin(["Uniform", "Visual-BiLSTM", "CLIP-zero-shot", "Concept-Adapter-MMR"])][
        ["video", "query_id", "method", "selected_indices"]
    ]
    nested = pd.read_csv(NESTED)[["video", "query_id", "method", "selected_indices"]]
    chan = pd.read_csv(CHAN)
    chan = chan[chan.method.eq("CHAN-MMR")][["video", "query_id", "method", "selected_indices"]]
    selected = pd.concat([selected, chan, nested], ignore_index=True)

    rows = []
    for row in selected.itertuples():
        video_queries = meta[meta.video == row.video].sort_values("query_id")
        labels = tags[int(row.video[-1]) - 1]
        indices = parse_indices(row.selected_indices)
        scores = []
        for query in video_queries.itertuples():
            score_or, score_and = relevance(
                indices, labels, concept_index[query.concept1], concept_index[query.concept2]
            )
            scores.append((query.query_id, score_or, score_and))
        true_or = next(value for query_id, value, _ in scores if query_id == row.query_id)
        true_and = next(value for query_id, _, value in scores if query_id == row.query_id)
        wrong_or = np.array([value for query_id, value, _ in scores if query_id != row.query_id])
        wrong_and = np.array([value for query_id, _, value in scores if query_id != row.query_id])
        rows.append({
            "video": row.video, "query_id": row.query_id, "method": row.method,
            "true_or": true_or, "wrong_mean_or": wrong_or.mean(),
            "margin_or": true_or - wrong_or.mean(), "discrimination_auc_or": auc_against_wrong(true_or, wrong_or),
            "true_and": true_and, "wrong_mean_and": wrong_and.mean(),
            "margin_and": true_and - wrong_and.mean(), "discrimination_auc_and": auc_against_wrong(true_and, wrong_and),
        })
    frame = pd.DataFrame(rows)
    for column in ["margin_or", "margin_and", "discrimination_auc_or", "discrimination_auc_and"]:
        target = 0.5 if column.startswith("discrimination") else 0.0
        frame.loc[np.isclose(frame[column], target, atol=1e-12), column] = target
    frame.to_csv(OUT, index=False, encoding="utf-8-sig")

    summary_rows = []
    for method, group in frame.groupby("method"):
        record = {"method": method, "n_queries": len(group), "n_videos": group.video.nunique()}
        for column in ["margin_or", "discrimination_auc_or", "margin_and", "discrimination_auc_and"]:
            low, high = cluster_bootstrap(group, column)
            record[f"{column}_mean"] = group[column].mean()
            record[f"{column}_cluster_ci_low"] = low
            record[f"{column}_cluster_ci_high"] = high
        for column in ["discrimination_auc_or", "discrimination_auc_and"]:
            differences = (group[column] - 0.5).to_numpy()
            differences[np.isclose(differences, 0.0, atol=1e-12)] = 0.0
            record[f"{column}_query_wilcoxon_p_exploratory"] = (
                1.0 if np.all(differences == 0) else float(wilcoxon(differences, zero_method="zsplit").pvalue)
            )
        summary_rows.append(record)
    summary = pd.DataFrame(summary_rows).sort_values("discrimination_auc_or_mean", ascending=False)
    summary.to_csv(SUMMARY, index=False, encoding="utf-8-sig")

    pairwise_rows = []
    metrics = ["margin_or", "discrimination_auc_or", "margin_and", "discrimination_auc_and"]
    ours = frame[frame.method == "Ours-Nested-Intent-Selector"].set_index(["video", "query_id"])
    for metric in metrics:
        family_rows, family_ps = [], []
        for baseline in ["Uniform", "Visual-BiLSTM", "CLIP-zero-shot", "Concept-Adapter-MMR", "CHAN-MMR"]:
            other = frame[frame.method == baseline].set_index(["video", "query_id"])
            differences = (ours[metric] - other[metric]).rename("difference").reset_index()
            low, high = cluster_bootstrap(differences, "difference")
            values = differences.difference.to_numpy()
            p_value = float(wilcoxon(values, zero_method="zsplit").pvalue)
            family_rows.append({
                "metric": metric, "comparison": f"Ours - {baseline}",
                "n_query_pairs": len(values), "mean_difference": values.mean(),
                "video_cluster_ci_low": low, "video_cluster_ci_high": high,
                "query_wilcoxon_p_exploratory": p_value,
            })
            family_ps.append(p_value)
        for row, adjusted in zip(family_rows, holm_adjust(family_ps)):
            row["holm_p_exploratory"] = adjusted
            pairwise_rows.append(row)
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise.to_csv(PAIRWISE, index=False, encoding="utf-8-sig")

    display = summary[[
        "method", "n_queries", "margin_or_mean", "discrimination_auc_or_mean",
        "margin_and_mean", "discrimination_auc_and_mean",
    ]]
    report = [
        "# QFVS查询置换负对照", "",
        "每个预测摘要均与其真实查询及同视频其余44个错误查询比较。",
        "区分AUC表示真实查询得分高于随机错误查询的概率，平分计0.5。", "",
        display.to_markdown(index=False, floatfmt=".4f"), "",
        "## 本文方法的配对差异", "",
        pairwise.to_markdown(index=False, floatfmt=".4f"), "",
        "置信区间以4个视频为聚类单位Bootstrap。查询级Wilcoxon p值仅作探索性结果，",
        "不能把180个查询视为180个完全独立视频样本。", "",
        "该分析是标签支撑的查询置换负对照，不替代QFVS官方F1，也不与其他指标事后加权。",
    ]
    REPORT.write_text("\n".join(report), "utf-8")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
