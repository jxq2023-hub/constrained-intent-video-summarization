"""Unified local comparison including public QFVS architectures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research_q3" / "results"
BASE = RESULTS / "qfvs_results.csv"
NESTED = RESULTS / "qfvs_nested_selection_results.csv"
CHAN = RESULTS / "qfvs_chan_fusion_results.csv"
FCSNA = RESULTS / "fcsna_qfvs_reproduction_results.csv"
TABLE_OUT = RESULTS / "qfvs_public_method_comparison.csv"
PAIRWISE_OUT = RESULTS / "qfvs_public_method_pairwise.csv"
REPORT_OUT = RESULTS / "qfvs_public_method_comparison.md"
SEED = 20260720
METRICS = ["fscore", "exact_fscore", "query_relevance_or", "query_relevance_and"]


def cluster_ci(joined: pd.DataFrame, left: str, right: str, iterations: int = 20_000):
    rng = np.random.default_rng(SEED)
    difference = (joined[left] - joined[right]).rename("difference").reset_index()
    means = difference.groupby("video").difference.mean().to_numpy(float)
    draws = rng.choice(means, size=(iterations, len(means)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    required = [BASE, NESTED, CHAN, FCSNA]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("run all comparison methods first: " + ", ".join(missing))
    base = pd.read_csv(BASE)
    base = base[base.method.isin(["Uniform", "Visual-BiLSTM", "CLIP-zero-shot", "Concept-Adapter-MMR"])]
    nested = pd.read_csv(NESTED)
    chan = pd.read_csv(CHAN)
    chan = chan[chan.method.isin(["CHAN", "CHAN-MMR"])]
    fcsna = pd.read_csv(FCSNA)
    frame = pd.concat([base, nested, chan, fcsna], ignore_index=True, sort=False)
    frame = frame.drop_duplicates(["method", "video", "query_id"], keep="last")
    expected = frame.groupby("method").size()
    if (expected != 180).any():
        raise RuntimeError(f"comparison requires 180 query units per method, found {expected.to_dict()}")

    summary = frame.groupby("method", as_index=False).agg(n=("query_id", "size"), **{
        metric: (metric, "mean") for metric in METRICS
    })
    order = ["Uniform", "Visual-BiLSTM", "CLIP-zero-shot", "CHAN", "CHAN-MMR",
             "FCSNA-QFVS-Controlled-Reproduction", "Concept-Adapter-MMR", "Ours-Nested-Intent-Selector"]
    summary["order"] = summary.method.map({name: index for index, name in enumerate(order)})
    summary = summary.sort_values("order").drop(columns="order")
    summary.to_csv(TABLE_OUT, index=False, encoding="utf-8-sig")

    units = frame.set_index(["method", "video", "query_id"])
    ours = units.loc["Ours-Nested-Intent-Selector"]
    rows = []
    for baseline in [name for name in order if name != "Ours-Nested-Intent-Selector"]:
        other = units.loc[baseline]
        joined = ours[METRICS].join(other[METRICS], lsuffix="_ours", rsuffix="_baseline", how="inner")
        for metric in METRICS:
            left, right = f"{metric}_ours", f"{metric}_baseline"
            differences = (joined[left] - joined[right]).to_numpy(float)
            low, high = cluster_ci(joined, left, right)
            try:
                p_value = float(wilcoxon(differences, zero_method="zsplit").pvalue)
            except ValueError:
                p_value = 1.0
            rows.append({
                "comparison": f"Ours - {baseline}", "metric": metric,
                "n_query_pairs": len(differences), "n_video_clusters": 4,
                "mean_difference": differences.mean(), "video_cluster_ci_low": low,
                "video_cluster_ci_high": high, "wilcoxon_p_exploratory": p_value,
            })
    pairwise = pd.DataFrame(rows)
    pairwise.to_csv(PAIRWISE_OUT, index=False, encoding="utf-8-sig")
    fcsna_focus = pairwise[pairwise.comparison == "Ours - FCSNA-QFVS-Controlled-Reproduction"]
    REPORT_OUT.write_text("\n".join([
        "# QFVS公开方法同协议对比", "",
        "全部方法使用相同180个查询—视频单元、2%镜头预算和本地语义匹配实现。",
        "FCSNA为论文定义与公开代码联合重实现；固定20个epoch，测试折不参与模型选择。", "",
        "## 主表", "", summary.to_markdown(index=False, floatfmt=".4f"), "",
        "## Ours与FCSNA配对差异", "", fcsna_focus.to_markdown(index=False, floatfmt=".4f"), "",
        "95%区间按4个源视频聚类Bootstrap；查询层Wilcoxon仅作探索性诊断。",
    ]), "utf-8")
    print(summary.to_string(index=False))
    print(fcsna_focus.to_string(index=False))


if __name__ == "__main__":
    main()
