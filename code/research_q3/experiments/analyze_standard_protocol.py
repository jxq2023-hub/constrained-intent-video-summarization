"""Aggregate standard-protocol results with paired statistics."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research_q3" / "results"
INPUT = RESULTS / "standard_protocol_strict_results.csv"


def bootstrap_ci(values: np.ndarray, seed: int = 20260717) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def main() -> None:
    frame = pd.read_csv(INPUT)
    expected = {"TVSum": 50, "SumMe": 25}
    for dataset, count in expected.items():
        subset = frame[frame.dataset == dataset]
        if subset.video_key.nunique() != count or len(subset) != count * 4:
            raise RuntimeError(f"incomplete {dataset} result: {len(subset)} rows")

    summary_rows = []
    test_rows = []
    for (dataset, method), group in frame.groupby(["dataset", "method"]):
        values = group.fscore.to_numpy(float)
        low, high = bootstrap_ci(values)
        summary_rows.append({
            "dataset": dataset, "method": method, "n": len(values),
            "fscore_mean": values.mean(), "fscore_std": values.std(ddof=1),
            "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
            "selected_ratio_mean": group.selected_ratio.mean(),
        })

    ours = "Ours-Adaptive"
    for dataset in expected:
        pivot = frame[frame.dataset == dataset].pivot(index="video_key", columns="method", values="fscore")
        for baseline in [c for c in pivot if c != ours]:
            diff = pivot[ours].to_numpy() - pivot[baseline].to_numpy()
            t = ttest_rel(pivot[ours], pivot[baseline])
            try:
                w = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                wp = float(w.pvalue)
            except ValueError:
                wp = 1.0
            test_rows.append({
                "dataset": dataset, "comparison": f"{ours} - {baseline}",
                "n": len(diff), "mean_difference": diff.mean(),
                "cohen_dz": diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else 0.0,
                "paired_t_p": float(t.pvalue), "wilcoxon_p": wp,
            })

    pd.DataFrame(summary_rows).to_csv(RESULTS / "standard_protocol_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(test_rows).to_csv(RESULTS / "standard_protocol_pairwise.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(summary_rows)
    tests = pd.DataFrame(test_rows)
    lines = [
        "# Standard TVSum/SumMe protocol results", "",
        "> TVSum Visual-BiLSTM predictions use non-overlapping five-fold evaluation",
        "> with three fixed seeds. SumMe is a cross-dataset transfer experiment:",
        "> the visual model was trained on all TVSum videos and never on SumMe.", "",
    ]
    for dataset in expected:
        lines += [f"## {dataset}", "", "| Method | F-score mean ± SD | 95% bootstrap CI | Budget |", "|---|---:|---:|---:|"]
        for row in summary[summary.dataset == dataset].sort_values("fscore_mean", ascending=False).itertuples():
            lines.append(
                f"| {row.method} | {row.fscore_mean:.4f} ± {row.fscore_std:.4f} | "
                f"[{row.bootstrap_ci95_low:.4f}, {row.bootstrap_ci95_high:.4f}] | {row.selected_ratio_mean:.4f} |"
            )
        lines += ["", "Paired comparisons (Ours minus baseline):", "", "| Comparison | Δ | Cohen dz | paired t p | Wilcoxon p |", "|---|---:|---:|---:|---:|"]
        for row in tests[tests.dataset == dataset].itertuples():
            lines.append(f"| {row.comparison} | {row.mean_difference:+.4f} | {row.cohen_dz:+.3f} | {row.paired_t_p:.4g} | {row.wilcoxon_p:.4g} |")
        lines.append("")
    (RESULTS / "standard_protocol_report.md").write_text("\n".join(lines), "utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
