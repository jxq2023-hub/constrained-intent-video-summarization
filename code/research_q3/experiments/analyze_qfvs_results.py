"""Statistical audit of QFVS official and query-grounded metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research_q3" / "results"
INPUT = RESULTS / "qfvs_results.csv"
NESTED_INPUT = RESULTS / "qfvs_nested_selection_results.csv"
OURS = "Ours-Nested-Intent-Selector"
METRICS = ["fscore", "exact_fscore", "query_relevance_or", "query_relevance_and"]


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def cluster_bootstrap_difference(
    frame: pd.DataFrame, metric: str, baseline: str, seed: int, n_boot: int = 20000,
) -> tuple[float, float]:
    """Bootstrap videos, retaining all queries inside each sampled video."""
    pivot = frame.pivot(index=["video", "query_id"], columns="method", values=metric)
    videos = np.asarray(sorted(frame.video.unique()))
    by_video = {
        video: (pivot.xs(video, level="video")[OURS] - pivot.xs(video, level="video")[baseline]).mean()
        for video in videos
    }
    rng = np.random.default_rng(seed)
    sampled = rng.choice(videos, size=(n_boot, len(videos)), replace=True)
    means = np.asarray([[by_video[v] for v in row] for row in sampled]).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def main() -> None:
    frame = pd.read_csv(INPUT)
    if NESTED_INPUT.exists():
        nested = pd.read_csv(NESTED_INPUT)
        shared = [column for column in frame.columns if column in nested.columns]
        frame = pd.concat([frame, nested[shared]], ignore_index=True, sort=False)
    expected_methods = sorted(frame.method.unique())
    expected_rows = frame[["video", "query_id"]].drop_duplicates().shape[0] * len(expected_methods)
    if len(frame) != expected_rows:
        raise RuntimeError(f"QFVS results are incomplete: {len(frame)} != {expected_rows}")
    if OURS not in expected_methods:
        raise RuntimeError(f"missing method: {OURS}")

    summary_rows: list[dict] = []
    for metric_index, metric in enumerate(METRICS):
        for method, group in frame.groupby("method"):
            values = group[metric].to_numpy(float)
            low, high = bootstrap_ci(values, 20260717 + metric_index)
            summary_rows.append({
                "metric": metric, "method": method, "n_queries": len(values),
                "mean": values.mean(), "std": values.std(ddof=1),
                "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
            })

    test_rows: list[dict] = []
    baselines = [method for method in expected_methods if method != OURS]
    for metric_index, metric in enumerate(METRICS):
        pivot = frame.pivot(index=["video", "query_id"], columns="method", values=metric)
        metric_rows = []
        raw_ps = []
        for baseline_index, baseline in enumerate(baselines):
            diff = (pivot[OURS] - pivot[baseline]).to_numpy(float)
            t_result = ttest_rel(pivot[OURS], pivot[baseline])
            try:
                wilcoxon_p = float(wilcoxon(diff, zero_method="wilcox").pvalue)
            except ValueError:
                wilcoxon_p = 1.0
            ci_low, ci_high = cluster_bootstrap_difference(
                frame, metric, baseline, 20260717 + metric_index * 100 + baseline_index,
            )
            row = {
                "metric": metric, "comparison": f"{OURS} - {baseline}",
                "n_queries": len(diff), "mean_difference": diff.mean(),
                "cohen_dz": diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else 0.0,
                "video_cluster_ci95_low": ci_low, "video_cluster_ci95_high": ci_high,
                "paired_t_p": float(t_result.pvalue), "wilcoxon_p": wilcoxon_p,
            }
            metric_rows.append(row)
            raw_ps.append(row["paired_t_p"])
        adjusted = holm_adjust(raw_ps)
        for row, p_adjusted in zip(metric_rows, adjusted):
            row["paired_t_holm_p"] = p_adjusted
            test_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    tests = pd.DataFrame(test_rows)
    summary.to_csv(RESULTS / "qfvs_metric_summary.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(RESULTS / "qfvs_pairwise_statistics.csv", index=False, encoding="utf-8-sig")

    labels = {
        "fscore": "Official semantic-matching F1",
        "exact_fscore": "Exact oracle-overlap F1",
        "query_relevance_or": "Query relevance (either concept)",
        "query_relevance_and": "Query relevance (both concepts)",
    }
    lines = [
        "# QFVS metric audit", "",
        "> All four metrics are reported. Official semantic matching is retained;",
        "> tag-grounded relevance and exact overlap are complementary diagnostics.",
        "> Confidence intervals for pairwise differences resample the four videos",
        "> as clusters. Holm correction is applied within each metric.", "",
    ]
    for metric in METRICS:
        lines += [f"## {labels[metric]}", "", "| Method | Mean ± SD | 95% query bootstrap CI |", "|---|---:|---:|"]
        part = summary[summary.metric == metric].sort_values("mean", ascending=False)
        for row in part.itertuples():
            lines.append(
                f"| {row.method} | {row.mean:.4f} ± {row.std:.4f} | "
                f"[{row.bootstrap_ci95_low:.4f}, {row.bootstrap_ci95_high:.4f}] |"
            )
        lines += ["", "| Ours minus baseline | Δ | Cohen dz | Video-cluster 95% CI | Holm p |", "|---|---:|---:|---:|---:|"]
        for row in tests[tests.metric == metric].itertuples():
            lines.append(
                f"| {row.comparison} | {row.mean_difference:+.4f} | {row.cohen_dz:+.3f} | "
                f"[{row.video_cluster_ci95_low:+.4f}, {row.video_cluster_ci95_high:+.4f}] | "
                f"{row.paired_t_holm_p:.4g} |"
            )
        lines.append("")
    report = "\n".join(lines)
    (RESULTS / "qfvs_metric_audit.md").write_text(report, "utf-8")
    print(report)


if __name__ == "__main__":
    main()
