"""Exploratory rater-level analysis for the completed five-rater blind study.

The independent unit for inferential comparisons is the rater. Each rater first
contributes one mean score per method and dimension across the 20 queries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, t, wilcoxon


ROOT = Path(__file__).resolve().parents[2]
INPUT = (
    ROOT
    / "research_q3"
    / "human_evaluation_final"
    / "responses"
    / "final_human_ratings_long.csv"
)
OUTPUT = ROOT / "research_q3" / "results"
METHODS = [
    "Uniform",
    "Visual-BiLSTM",
    "CLIP-zero-shot",
    "Ours-Nested-Intent-Selector",
]
OURS = "Ours-Nested-Intent-Selector"
DIMENSIONS = {
    "relevance_1_5": "意图相关性",
    "both_concepts_coverage_1_5": "双概念覆盖",
    "non_redundancy_1_5": "非冗余性",
    "overall_usefulness_1_5": "总体可用性",
}


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def icc_two_way_absolute(matrix: np.ndarray) -> tuple[float, float]:
    """Return ICC(2,1) and ICC(2,k), two-way random absolute agreement."""
    matrix = np.asarray(matrix, dtype=float)
    n, k = matrix.shape
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ms_rows = k * np.square(row_means - grand).sum() / (n - 1)
    ms_cols = n * np.square(col_means - grand).sum() / (k - 1)
    residual = matrix - row_means[:, None] - col_means[None, :] + grand
    ms_error = np.square(residual).sum() / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    single = (ms_rows - ms_error) / denominator if denominator else np.nan
    average = (
        (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n)
        if ms_rows
        else np.nan
    )
    return float(single), float(average)


def mean_ci(values: pd.Series) -> tuple[float, float]:
    values = values.astype(float)
    mean = float(values.mean())
    if len(values) < 2:
        return np.nan, np.nan
    half_width = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean - half_width, mean + half_width


def main() -> None:
    ratings = pd.read_csv(INPUT)
    if ratings.rater_id.nunique() != 5:
        raise ValueError(f"expected five raters, found {ratings.rater_id.nunique()}")
    if set(ratings.method) != set(METHODS):
        raise ValueError(f"method mismatch: {sorted(set(ratings.method))}")
    if len(ratings) != 400:
        raise ValueError(f"expected 400 rater-sample rows, found {len(ratings)}")

    rater_means = (
        ratings.groupby(["rater_id", "method"])[list(DIMENSIONS)]
        .mean()
        .reset_index()
    )
    rater_means["four_dimension_mean"] = rater_means[list(DIMENSIONS)].mean(axis=1)
    rater_means.to_csv(
        OUTPUT / "final_human_5rater_rater_means.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows: list[dict[str, object]] = []
    for column, label in {**DIMENSIONS, "four_dimension_mean": "四维平均"}.items():
        for method in METHODS:
            values = rater_means.loc[rater_means.method == method, column]
            low, high = mean_ci(values)
            summary_rows.append(
                {
                    "dimension": column,
                    "dimension_zh": label,
                    "method": method,
                    "n_raters": len(values),
                    "mean": values.mean(),
                    "sd_between_raters": values.std(ddof=1),
                    "ci95_low_t": low,
                    "ci95_high_t": high,
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT / "final_human_5rater_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    omnibus_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    for column, label in {**DIMENSIONS, "four_dimension_mean": "四维平均"}.items():
        matrix = rater_means.pivot(index="rater_id", columns="method", values=column)[METHODS]
        statistic, p_value = friedmanchisquare(*[matrix[method] for method in METHODS])
        omnibus_rows.append(
            {
                "dimension": column,
                "dimension_zh": label,
                "n_raters": len(matrix),
                "friedman_chi2": statistic,
                "p_value": p_value,
            }
        )
        current_rows: list[dict[str, object]] = []
        current_p: list[float] = []
        for baseline in [method for method in METHODS if method != OURS]:
            difference = matrix[OURS] - matrix[baseline]
            result = wilcoxon(difference, alternative="two-sided", method="exact")
            low, high = mean_ci(difference)
            current_rows.append(
                {
                    "dimension": column,
                    "dimension_zh": label,
                    "comparison": f"{OURS} - {baseline}",
                    "n_raters": len(difference),
                    "mean_difference": difference.mean(),
                    "sd_difference": difference.std(ddof=1),
                    "ci95_low_t": low,
                    "ci95_high_t": high,
                    "raw_p": float(result.pvalue),
                }
            )
            current_p.append(float(result.pvalue))
        for row, adjusted in zip(current_rows, holm_adjust(current_p)):
            row["holm_p"] = adjusted
            pairwise_rows.append(row)

    omnibus = pd.DataFrame(omnibus_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    omnibus.to_csv(
        OUTPUT / "final_human_5rater_omnibus.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pairwise.to_csv(
        OUTPUT / "final_human_5rater_pairwise.csv",
        index=False,
        encoding="utf-8-sig",
    )

    agreement_rows: list[dict[str, object]] = []
    for column, label in DIMENSIONS.items():
        matrix = ratings.pivot_table(
            index=["query_number", "method"],
            columns="rater_id",
            values=column,
            aggfunc="first",
        )
        if matrix.isna().any().any() or matrix.shape != (80, 5):
            raise ValueError(f"unbalanced agreement matrix for {column}: {matrix.shape}")
        single, average = icc_two_way_absolute(matrix.to_numpy())
        agreement_rows.append(
            {
                "dimension": column,
                "dimension_zh": label,
                "n_stimuli": matrix.shape[0],
                "n_raters": matrix.shape[1],
                "icc_2_1": single,
                "icc_2_k": average,
            }
        )
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(
        OUTPUT / "final_human_5rater_agreement.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ranks = (
        rater_means.pivot(
            index="rater_id",
            columns="method",
            values="four_dimension_mean",
        )
        .rank(axis=1, ascending=False, method="min")
        .reset_index()
    )
    ranks.to_csv(
        OUTPUT / "final_human_5rater_ranks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = [
        "# 五名评价者盲评结果（探索性）",
        "",
        "- 独立评价者：5名；每名评价者完成20个查询×4种匿名方法×4个五点评分维度。",
        "- 原始评分：1600项；缺失值与越界值均为0。",
        "- 推断统计以评价者为独立单位；每名评价者先对20个查询求方法均值。",
        "- 样本量较小，方法两两比较的最小双侧精确Wilcoxon p值为0.0625，因此不作单个基线的确认性显著声明。",
        "",
        "## 评价者层面描述性结果",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Friedman总体检验",
        "",
        omnibus.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 本文方法与基线的评价者层面配对比较",
        "",
        pairwise.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 评价者间一致性",
        "",
        agreement.to_markdown(index=False, floatfmt=".4f"),
        "",
        "所有评价者均把本文方法的四维平均评分排在第一位。"
        "配对差的95%区间为基于5名评价者均值的t区间，仅作描述性不确定性展示。",
    ]
    (OUTPUT / "final_human_5rater_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(omnibus.to_string(index=False))
    print(agreement.to_string(index=False))


if __name__ == "__main__":
    main()
