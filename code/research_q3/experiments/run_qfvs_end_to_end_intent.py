"""End-to-end QFVS experiment from natural-language intent to final summary.

The experiment keeps the final nested selector fixed and changes only how the
two query concepts are obtained:

* Oracle-Intent: official QFVS concept pair (upper bound for intent parsing);
* LLM-Parsed: cached unseen-template output from the production IntentParser;
* Rule-Parsed: deterministic dictionary extraction from the raw request;
* No-Intent-Visual: query-independent Visual-BiLSTM summary.

No prompt, selector hyperparameter, or metric is tuned in this script.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge

from research_q3.experiments.run_qfvs_nested_selection import combine, minmax
from research_q3.methods.diverse_selection import mmr_select
from research_q3.protocols.qfvs_semantic import (
    evaluate_exact_and_query_relevance,
    evaluate_semantic_summary,
    load_qfvs_tags,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research_q3" / "data" / "qfvs"
CACHE = DATA / "feature_cache"
ANNOTATIONS = DATA / "annotations"
RESULTS = ROOT / "research_q3" / "results"
CONCEPT_PATH = ROOT / "research_q3" / "configs" / "qfvs_concepts.json"
INTENT_RAW = RESULTS / "intent_robustness_post_raw.jsonl"
NESTED = RESULTS / "qfvs_nested_selection_results.csv"
BASE_RESULTS = RESULTS / "qfvs_results.csv"
RAW_OUT = RESULTS / "qfvs_end_to_end_intent_results.csv"
SUMMARY_OUT = RESULTS / "qfvs_end_to_end_intent_summary.csv"
PAIRWISE_OUT = RESULTS / "qfvs_end_to_end_intent_pairwise.csv"
REPORT_OUT = RESULTS / "qfvs_end_to_end_intent_report.md"
SEED = 20260718
METRICS = ["fscore", "exact_fscore", "query_relevance_or", "query_relevance_and"]


def normalize(text: str) -> str:
    return re.sub(r"[\s、，,。和与或/\-]", "", str(text).lower())


def concept_aliases(translation: str) -> list[str]:
    values = {translation}
    values.update(part for part in re.split(r"或|/", translation) if part)
    values.update({translation.replace("人物", ""), translation.replace("东西", "")})
    return sorted({normalize(value) for value in values if normalize(value)}, key=len, reverse=True)


def map_focus(focus: list[str], translations: dict[str, str]) -> list[str]:
    """Map parser focus phrases to canonical concepts without using ground truth."""
    mapped: list[str] = []
    for item in focus:
        value = normalize(item)
        if not value:
            continue
        candidates = []
        for concept, translation in translations.items():
            for alias in concept_aliases(translation):
                exact = int(value == alias)
                contained = int(alias in value or value in alias)
                if contained:
                    candidates.append((exact, len(alias), concept))
        if candidates:
            concept = max(candidates)[2]
            if concept not in mapped:
                mapped.append(concept)
    return mapped[:2]


def rule_parse(query: str, translations: dict[str, str]) -> list[str]:
    """Longest non-overlapping dictionary match over the raw request."""
    text = normalize(query)
    matches: list[tuple[int, int, int, str]] = []
    for concept, translation in translations.items():
        for alias in concept_aliases(translation):
            start = text.find(alias)
            while start >= 0:
                matches.append((len(alias), start, start + len(alias), concept))
                start = text.find(alias, start + 1)
    occupied: list[tuple[int, int]] = []
    accepted: list[tuple[int, str]] = []
    for _, start, end, concept in sorted(matches, key=lambda item: (-item[0], item[1], item[3])):
        if any(not (end <= left or start >= right) for left, right in occupied):
            continue
        if concept in {value for _, value in accepted}:
            continue
        occupied.append((start, end))
        accepted.append((start, concept))
    return [concept for _, concept in sorted(accepted)[:2]]


def parse_config(text: str) -> tuple[str, float, float]:
    parts = dict(item.split("=") for item in text.split(";"))
    return parts["rule"], float(parts["visual"]), float(parts["diversity"])


def load_intent_cases() -> list[dict]:
    records = [json.loads(line) for line in INTENT_RAW.read_text("utf-8").splitlines() if line.strip()]
    if len(records) != 540:
        raise RuntimeError(f"expected 540 cached unseen-template requests, found {len(records)}")
    return records


def bootstrap_ci(values: np.ndarray, iterations: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    translations: dict[str, str] = json.loads(CONCEPT_PATH.read_text("utf-8"))
    concept_names = list(translations)
    concept_index = {name: index for index, name in enumerate(concept_names)}
    tags = load_qfvs_tags(str(ANNOTATIONS / "Tags.mat"))
    videos = ["P01", "P02", "P03", "P04"]

    nested = pd.read_csv(NESTED)
    configs = {video: parse_config(group.config.iloc[0]) for video, group in nested.groupby("video")}
    base_frame = pd.read_csv(BASE_RESULTS)
    visual_rows = base_frame[base_frame.method == "Visual-BiLSTM"].set_index(["video", "query_id"])

    cached: dict[str, dict[str, np.ndarray]] = {}
    for video in videos:
        base = np.load(CACHE / f"{video}.npz")
        cached[video] = {
            "visual": minmax(base["visual_scores"]),
            "image": np.load(CACHE / f"{video}_openai_clip.npz")["image_features"].astype(np.float32),
        }

    calibrated: dict[str, np.ndarray] = {}
    for video in videos:
        train = [name for name in videos if name != video]
        train_x = np.concatenate([cached[name]["image"] for name in train])
        train_y = np.concatenate([tags[int(name[-1]) - 1] for name in train]).astype(np.float32)
        head = Ridge(alpha=10.0, fit_intercept=True, solver="lsqr").fit(train_x, train_y)
        calibrated[video] = head.predict(cached[video]["image"])

    selection_cache: dict[tuple[str, str, str], list[int]] = {}
    evaluation_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, float]] = {}

    def select(video: str, concept1: str, concept2: str) -> list[int]:
        key = (video, concept1, concept2)
        if key in selection_cache:
            return selection_cache[key]
        rule, visual_weight, diversity = configs[video]
        image = cached[video]["image"]
        visual = cached[video]["visual"]
        semantic = combine(
            calibrated[video][:, concept_index[concept1]],
            calibrated[video][:, concept_index[concept2]],
            rule,
        )
        score = (1.0 - visual_weight) * semantic + visual_weight * visual
        k = max(1, int(len(score) * 0.02))
        chosen = (
            mmr_select(score, image, k, diversity=diversity)
            if diversity > 0
            else np.argsort(-score, kind="stable")[:k]
        )
        selection_cache[key] = [int(value) for value in chosen]
        return selection_cache[key]

    def evaluate(case: dict, selected: list[int]) -> dict[str, float]:
        key = (case["video"], case["query_id"], tuple(selected))
        if key in evaluation_cache:
            return evaluation_cache[key]
        oracle_path = ANNOTATIONS / "Oracle_summaries" / case["video"] / f"{case['query_id']}_oracle.txt"
        oracle = [int(line.strip()) - 1 for line in oracle_path.read_text("utf-8").splitlines() if line.strip()]
        video_index = int(case["video"][-1]) - 1
        c1, c2 = concept_index[case["concept1"]], concept_index[case["concept2"]]
        official = evaluate_semantic_summary(selected, oracle, tags[video_index])
        auxiliary = evaluate_exact_and_query_relevance(selected, oracle, tags[video_index], c1, c2)
        evaluation_cache[key] = {**official, **auxiliary}
        return evaluation_cache[key]

    rows: list[dict] = []
    for number, case in enumerate(load_intent_cases(), start=1):
        llm_concepts = map_focus([str(value) for value in case.get("focus", [])], translations)
        rule_concepts = rule_parse(case["query"], translations)
        methods = {
            "Oracle-Intent": [case["concept1"], case["concept2"]],
            "LLM-Parsed": llm_concepts if case.get("goal") == "semantic_extract" else [],
            "Rule-Parsed": rule_concepts,
            "No-Intent-Visual": [],
        }
        true_concepts = {case["concept1"], case["concept2"]}
        for method, parsed in methods.items():
            routeable = len(parsed) == 2
            parser_success = routeable and set(parsed) == true_concepts
            if method == "No-Intent-Visual" or not routeable:
                selected = json.loads(visual_rows.loc[(case["video"], case["query_id"]), "selected_indices"])
                route = "visual_fallback"
            else:
                selected = select(case["video"], parsed[0], parsed[1])
                route = "intent_selector"
            scores = evaluate(case, [int(value) for value in selected])
            rows.append({
                "case_id": case["case_id"], "video": case["video"], "query_id": case["query_id"],
                "template": case["template"], "query": case["query"], "method": method,
                "true_concept1": case["concept1"], "true_concept2": case["concept2"],
                "parsed_concepts": json.dumps(parsed, ensure_ascii=False),
                "routeable": routeable, "parser_success": parser_success, "route": route,
                "selected_indices": json.dumps(selected), **{metric: scores[metric] for metric in METRICS},
            })
        if number % 50 == 0:
            print(f"processed {number}/540 requests", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(RAW_OUT, index=False, encoding="utf-8-sig")
    summary = frame.groupby(["method", "template"], as_index=False).agg(
        n=("case_id", "size"), parser_success=("parser_success", "mean"),
        **{metric: (metric, "mean") for metric in METRICS},
    )
    overall = frame.groupby("method", as_index=False).agg(
        n=("case_id", "size"), parser_success=("parser_success", "mean"),
        **{metric: (metric, "mean") for metric in METRICS},
    )
    overall["template"] = "all"
    summary = pd.concat([summary, overall], ignore_index=True)
    summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

    query_units = frame.groupby(["video", "query_id", "method"], as_index=False)[METRICS].mean()
    pairwise_rows = []
    oracle = query_units[query_units.method == "Oracle-Intent"].set_index(["video", "query_id"])
    for method in ["LLM-Parsed", "Rule-Parsed", "No-Intent-Visual"]:
        other = query_units[query_units.method == method].set_index(["video", "query_id"])
        for metric in METRICS:
            differences = (other[metric] - oracle[metric]).to_numpy()
            low, high = bootstrap_ci(differences)
            try:
                p_value = float(wilcoxon(differences, zero_method="zsplit").pvalue)
            except ValueError:
                p_value = 1.0
            pairwise_rows.append({
                "comparison": f"{method} - Oracle-Intent", "metric": metric,
                "n_query_pairs": len(differences), "mean_difference": differences.mean(),
                "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                "wilcoxon_p_exploratory": p_value,
            })
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise.to_csv(PAIRWISE_OUT, index=False, encoding="utf-8-sig")

    main = summary[summary.template == "all"].copy()
    report = [
        "# QFVS端到端意图实验报告", "",
        "本实验固定最终嵌套选择器及其留一视频配置，仅改变自然语言意图的获得方式。",
        "LLM结果来自此前冻结的540条未见模板输出；解析失败统一回退到查询无关的Visual-BiLSTM摘要。",
        "Rule-Parsed使用预先定义的中文概念词典最长匹配。Oracle-Intent仅作为解析上界。", "",
        "## 总体结果", "",
        main[["method", "n", "parser_success", *METRICS]].to_markdown(index=False, floatfmt=".4f"), "",
        "## 分模板结果", "",
        summary[summary.template != "all"].to_markdown(index=False, floatfmt=".4f"), "",
        "## 与Oracle-Intent的配对差异", "",
        pairwise.to_markdown(index=False, floatfmt=".4f"), "",
        "统计说明：Wilcoxon检验以180个查询对为探索性配对单位；同一视频内查询并非完全独立，",
        "因此论文应同时报告效应量和Bootstrap区间，不把p值作为唯一证据。", "",
        "重要解释：若Rule-Parsed接近Oracle，说明当前请求模板中的概念词较明确，",
        "不能据此宣称LLM在实体抽取上优于规则；LLM的价值应结合开放表达、约束解析和策略路由讨论。",
    ]
    REPORT_OUT.write_text("\n".join(report), "utf-8")
    print(main[["method", "parser_success", *METRICS]].to_string(index=False))
    print(f"wrote {RAW_OUT}")


if __name__ == "__main__":
    main()
