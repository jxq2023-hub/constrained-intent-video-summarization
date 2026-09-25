"""Cluster-aware statistics for the simulated user-language stress test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research_q3" / "results"
INPUT = RESULTS / "qfvs_simulated_user_language_results.csv"
PAIRWISE_OUT = RESULTS / "qfvs_simulated_user_language_pairwise.csv"
PARSER_OUT = RESULTS / "qfvs_simulated_user_language_parser_ci.csv"
REPORT_OUT = RESULTS / "qfvs_simulated_user_language_statistics.md"
SEED = 20260720
METRICS = ["intent_correct", "fscore", "exact_fscore", "query_relevance_or", "query_relevance_and"]
COMPARISONS = [
    ("LLM-Ontology", "Rule-Canonical-Lexicon"),
    ("LLM-Ontology", "No-Intent-Visual"),
    ("LLM-Ontology", "Oracle-Intent"),
]


def cluster_ci(joined: pd.DataFrame, left: str, right: str, iterations: int = 20_000):
    rng = np.random.default_rng(SEED)
    differences = (joined[left] - joined[right]).rename("difference").reset_index()
    cluster_means = differences.groupby("video").difference.mean().to_numpy(float)
    draws = rng.choice(cluster_means, size=(iterations, len(cluster_means)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def wilson(values: pd.Series) -> dict:
    observed = values.dropna().astype(bool)
    count = int(observed.sum())
    n = len(observed)
    if n:
        z = 1.959963984540054
        p = count / n
        denominator = 1.0 + z * z / n
        center = (p + z * z / (2.0 * n)) / denominator
        half_width = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
        low, high = center - half_width, center + half_width
    else:
        low, high = np.nan, np.nan
    return {"n": n, "successes": count, "rate": count / n if n else np.nan, "ci_low": low, "ci_high": high}


def main() -> None:
    frame = pd.read_csv(INPUT)
    llm = frame[frame.method == "LLM-Ontology"].drop_duplicates("case_id")
    parser_rows = []
    for style, group in list(llm.groupby("style")) + [("all", llm)]:
        for metric in ["goal_correct", "routeable", "intent_correct"]:
            parser_rows.append({"style": style, "metric": metric, **wilson(group[metric])})
        constrained = group[group.constraint_count > 0]
        if len(constrained):
            parser_rows.append({"style": style, "metric": "constraint_correct", **wilson(constrained.constraint_correct)})
        formatted = group[group.format_correct.notna()]
        if len(formatted):
            parser_rows.append({"style": style, "metric": "format_correct", **wilson(formatted.format_correct)})
    parser_frame = pd.DataFrame(parser_rows)
    parser_frame.to_csv(PARSER_OUT, index=False, encoding="utf-8-sig")

    units = frame.groupby(["method", "video", "query_id"], as_index=False)[METRICS].mean()
    rows = []
    for method_a, method_b in COMPARISONS:
        a = units[units.method == method_a].set_index(["video", "query_id"])
        b = units[units.method == method_b].set_index(["video", "query_id"])
        joined = a[METRICS].join(b[METRICS], lsuffix="_a", rsuffix="_b", how="inner")
        for metric in METRICS:
            left, right = f"{metric}_a", f"{metric}_b"
            differences = (joined[left] - joined[right]).to_numpy(float)
            low, high = cluster_ci(joined, left, right)
            try:
                p_value = float(wilcoxon(differences, zero_method="zsplit").pvalue)
            except ValueError:
                p_value = 1.0
            rows.append({
                "comparison": f"{method_a} - {method_b}", "metric": metric,
                "n_query_pairs": len(differences), "n_video_clusters": 4,
                "mean_difference": differences.mean(), "video_cluster_ci_low": low,
                "video_cluster_ci_high": high, "wilcoxon_p_exploratory": p_value,
            })
    pairwise = pd.DataFrame(rows)
    pairwise.to_csv(PAIRWISE_OUT, index=False, encoding="utf-8-sig")
    focus = pairwise[pairwise.metric.isin(["intent_correct", "fscore", "query_relevance_or"])]
    REPORT_OUT.write_text("\n".join([
        "# 模拟用户式自然语言实验统计", "",
        "语料为研究者模拟而非真实用户请求。比例区间使用Wilson方法；方法差异先在同一查询的",
        "六种语言风格间求均值，再按四个源视频聚类Bootstrap。查询层Wilcoxon仅作探索性诊断。", "",
        "## 解析比例及区间", "", parser_frame.to_markdown(index=False, floatfmt=".4f"), "",
        "## 端到端配对差异", "", focus.to_markdown(index=False, floatfmt=".4f"),
    ]), "utf-8")
    print(parser_frame[parser_frame["style"] == "all"].to_string(index=False))
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
