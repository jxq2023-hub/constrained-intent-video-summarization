"""Researcher-simulated user-language stress test for QFVS.

The corpus is generated deterministically from the 180 official QFVS query
pairs.  It imitates common request styles but MUST NOT be described as real
user data.  The frozen nested selector and evaluation protocol are reused so
that only intent interpretation changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from agent.intent_parser import IntentParser
from research_q3.experiments.run_qfvs_end_to_end_intent import METRICS, map_focus, parse_config, rule_parse
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
CONFIG = ROOT / "research_q3" / "configs" / "qfvs_simulated_user_language.json"
TRANSLATIONS = ROOT / "research_q3" / "configs" / "qfvs_concepts.json"
EXPRESSIONS = ROOT / "research_q3" / "configs" / "qfvs_open_expression_concepts.json"
BASE_RESULTS = RESULTS / "qfvs_results.csv"
NESTED_RESULTS = RESULTS / "qfvs_nested_selection_results.csv"
CORPUS_OUT = RESULTS / "qfvs_simulated_user_language_corpus.jsonl"
LLM_OUT = RESULTS / "qfvs_simulated_user_language_llm_raw.jsonl"
RAW_OUT = RESULTS / "qfvs_simulated_user_language_results.csv"
SUMMARY_OUT = RESULTS / "qfvs_simulated_user_language_summary.csv"
REPORT_OUT = RESULTS / "qfvs_simulated_user_language_report.md"


def load_pairs() -> list[dict]:
    base = pd.read_csv(BASE_RESULTS)
    pairs = (
        base[["video", "query_id", "concept1", "concept2"]]
        .drop_duplicates()
        .sort_values(["video", "query_id"])
    )
    return pairs.to_dict("records")


def build_corpus(limit: int | None = None) -> list[dict]:
    spec = json.loads(CONFIG.read_text("utf-8"))
    expressions = json.loads(EXPRESSIONS.read_text("utf-8"))
    pairs = load_pairs()
    if limit is not None:
        pairs = pairs[:limit]
    cases: list[dict] = []
    for pair_index, pair in enumerate(pairs):
        for style in spec["styles"]:
            expression = style["expression"]
            if expression == "canonical_english":
                a, b = pair["concept1"], pair["concept2"]
            else:
                a = expressions[pair["concept1"]][expression]
                b = expressions[pair["concept2"]][expression]
            variant = style
            variant_index = None
            if "variants" in style:
                variant_index = pair_index % len(style["variants"])
                variant = {**style, **style["variants"][variant_index]}
            case_id = f"{pair['video']}:{pair['query_id']}:{style['name']}"
            cases.append({
                **pair,
                "case_id": case_id,
                "style": style["name"],
                "variant_index": variant_index,
                "query": variant["template"].format(a=a, b=b),
                "expected_goal": "semantic_extract",
                "expected_constraints": variant.get("expected_constraints", {}),
                "expected_output_format": variant.get("expected_output_format"),
                "corpus_status": spec["status"],
                "seed": spec["seed"],
            })
    CORPUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS_OUT.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    return cases


def parse_cases(cases: list[dict], fresh: bool) -> dict[str, dict]:
    if fresh and LLM_OUT.exists():
        LLM_OUT.unlink()
    existing: dict[str, dict] = {}
    if LLM_OUT.exists():
        for line in LLM_OUT.read_text("utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing[record["case_id"]] = record
    translations = json.loads(TRANSLATIONS.read_text("utf-8"))
    parser = IntentParser(temperature=0.0, focus_vocabulary=list(translations.values()))
    for index, case in enumerate(cases, start=1):
        if case["case_id"] in existing:
            continue
        intent = parser.parse(case["query"])
        record = {
            **case,
            "goal": intent.goal,
            "focus": intent.focus,
            "mapped_concepts": map_focus([str(value) for value in intent.focus], translations),
            "constraints": intent.constraints,
            "output_format": intent.output_format,
            "confidence": intent.confidence,
        }
        with LLM_OUT.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        existing[case["case_id"]] = record
        print(f"parsed {index}/{len(cases)} {case['case_id']} -> {intent.goal}/{record['mapped_concepts']}", flush=True)
    return existing


def constraint_score(expected: dict, actual: dict) -> tuple[bool | None, int]:
    if not expected:
        return None, 0
    correct = all(actual.get(key) == value for key, value in expected.items())
    return correct, len(expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="limit official query pairs before style expansion")
    parser.add_argument("--fresh", action="store_true", help="discard cached LLM outputs")
    args = parser.parse_args()

    cases = build_corpus(args.limit)
    parsed = parse_cases(cases, args.fresh)
    translations = json.loads(TRANSLATIONS.read_text("utf-8"))
    concepts = list(translations)
    concept_index = {name: index for index, name in enumerate(concepts)}
    tags = load_qfvs_tags(str(ANNOTATIONS / "Tags.mat"))
    videos = ["P01", "P02", "P03", "P04"]

    nested = pd.read_csv(NESTED_RESULTS)
    configs = {video: parse_config(group.config.iloc[0]) for video, group in nested.groupby("video")}
    base = pd.read_csv(BASE_RESULTS)
    visual_rows = base[base.method == "Visual-BiLSTM"].set_index(["video", "query_id"])
    cached = {}
    for video in videos:
        source = np.load(CACHE / f"{video}.npz")
        cached[video] = {
            "visual": minmax(source["visual_scores"]),
            "image": np.load(CACHE / f"{video}_openai_clip.npz")["image_features"].astype(np.float32),
        }
    calibrated = {}
    for video in videos:
        train = [name for name in videos if name != video]
        head = Ridge(alpha=10.0, fit_intercept=True, solver="lsqr").fit(
            np.concatenate([cached[name]["image"] for name in train]),
            np.concatenate([tags[int(name[-1]) - 1] for name in train]).astype(np.float32),
        )
        calibrated[video] = head.predict(cached[video]["image"])

    selection_cache: dict[tuple, list[int]] = {}
    evaluation_cache: dict[tuple, dict] = {}

    def select(video: str, chosen_concepts: list[str]) -> list[int]:
        key = (video, *chosen_concepts)
        if key in selection_cache:
            return selection_cache[key]
        rule, visual_weight, diversity = configs[video]
        semantic = combine(
            calibrated[video][:, concept_index[chosen_concepts[0]]],
            calibrated[video][:, concept_index[chosen_concepts[1]]],
            rule,
        )
        score = (1.0 - visual_weight) * semantic + visual_weight * cached[video]["visual"]
        k = max(1, int(len(score) * 0.02))
        chosen = (
            mmr_select(score, cached[video]["image"], k, diversity=diversity)
            if diversity > 0 else np.argsort(-score, kind="stable")[:k]
        )
        selection_cache[key] = [int(value) for value in chosen]
        return selection_cache[key]

    def evaluate(case: dict, selected: list[int]) -> dict:
        key = (case["video"], case["query_id"], tuple(selected))
        if key in evaluation_cache:
            return evaluation_cache[key]
        oracle_path = ANNOTATIONS / "Oracle_summaries" / case["video"] / f"{case['query_id']}_oracle.txt"
        oracle = [int(line) - 1 for line in oracle_path.read_text("utf-8").splitlines() if line.strip()]
        video_index = int(case["video"][-1]) - 1
        official = evaluate_semantic_summary(selected, oracle, tags[video_index])
        auxiliary = evaluate_exact_and_query_relevance(
            selected, oracle, tags[video_index],
            concept_index[case["concept1"]], concept_index[case["concept2"]],
        )
        evaluation_cache[key] = {**official, **auxiliary}
        return evaluation_cache[key]

    rows = []
    for index, case in enumerate(cases, start=1):
        record = parsed[case["case_id"]]
        llm_concepts = record["mapped_concepts"] if record["goal"] == "semantic_extract" else []
        rule_concepts = rule_parse(case["query"], translations)
        methods = {
            "Oracle-Intent": [case["concept1"], case["concept2"]],
            "LLM-Ontology": llm_concepts,
            "Rule-Canonical-Lexicon": rule_concepts,
            "No-Intent-Visual": [],
        }
        expected_set = {case["concept1"], case["concept2"]}
        constraint_correct, constraint_count = constraint_score(case["expected_constraints"], record["constraints"])
        format_correct = None if case["expected_output_format"] is None else record["output_format"] == case["expected_output_format"]
        for method, chosen_concepts in methods.items():
            routeable = len(chosen_concepts) == 2
            intent_correct = routeable and set(chosen_concepts) == expected_set
            if method == "No-Intent-Visual" or not routeable:
                selected = [int(value) for value in json.loads(
                    visual_rows.loc[(case["video"], case["query_id"]), "selected_indices"]
                )]
                route = "visual_fallback"
            else:
                selected = select(case["video"], chosen_concepts)
                route = "intent_selector"
            scores = evaluate(case, selected)
            rows.append({
                **case,
                "method": method,
                "parsed_goal": record["goal"],
                "parsed_focus": json.dumps(record["focus"], ensure_ascii=False),
                "parsed_concepts": json.dumps(chosen_concepts, ensure_ascii=False),
                "parsed_constraints": json.dumps(record["constraints"], ensure_ascii=False),
                "parsed_output_format": record["output_format"],
                "goal_correct": record["goal"] == case["expected_goal"],
                "routeable": routeable,
                "intent_correct": intent_correct,
                "constraint_correct": constraint_correct,
                "constraint_count": constraint_count,
                "format_correct": format_correct,
                "route": route,
                "selected_indices": json.dumps(selected),
                **{metric: scores[metric] for metric in METRICS},
            })
        if index % 50 == 0:
            print(f"evaluated {index}/{len(cases)}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(RAW_OUT, index=False, encoding="utf-8-sig")
    summary = frame.groupby(["method", "style"], as_index=False).agg(
        n=("case_id", "size"), goal_accuracy=("goal_correct", "mean"),
        routeable=("routeable", "mean"), intent_accuracy=("intent_correct", "mean"),
        **{metric: (metric, "mean") for metric in METRICS},
    )
    overall = frame.groupby("method", as_index=False).agg(
        n=("case_id", "size"), goal_accuracy=("goal_correct", "mean"),
        routeable=("routeable", "mean"), intent_accuracy=("intent_correct", "mean"),
        **{metric: (metric, "mean") for metric in METRICS},
    )
    overall["style"] = "all"
    summary = pd.concat([summary, overall], ignore_index=True)
    summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

    llm_rows = frame[frame.method == "LLM-Ontology"].drop_duplicates("case_id")
    constrained = llm_rows[llm_rows.constraint_count > 0]
    formatted = llm_rows[llm_rows.format_correct.notna()]
    parser_table = pd.DataFrame([{
        "n_requests": len(llm_rows),
        "goal_accuracy": llm_rows.goal_correct.mean(),
        "routeable": llm_rows.routeable.mean(),
        "exact_concept_pair_accuracy": llm_rows.intent_correct.mean(),
        "constraint_field_accuracy": constrained.constraint_correct.mean(),
        "n_constraint_requests": len(constrained),
        "output_format_accuracy": formatted.format_correct.mean(),
        "n_format_requests": len(formatted),
    }])
    main_table = summary[summary["style"] == "all"]
    report = [
        "# QFVS模拟用户式自然语言压力测试", "",
        "**数据边界：本语料由研究者按冻结模板和QFVS概念表确定性生成，不是真实用户数据。**",
        "它用于检验口语、省略、无标点、描述性表达、中英混用及显式约束下的稳健性，不能替代真实用户研究。", "",
        "## LLM解析结果", "", parser_table.to_markdown(index=False, floatfmt=".4f"), "",
        "## 端到端总体结果", "", main_table.to_markdown(index=False, floatfmt=".4f"), "",
        "## 分语言风格结果", "", summary[summary["style"] != "all"].to_markdown(index=False, floatfmt=".4f"), "",
        "所有方法共用相同2%摘要预算和冻结选择器。带时长/帧数的请求仅用于评价约束解析，",
        "不改变QFVS摘要预算，以避免不同方法因输出长度不同而失去可比性。",
    ]
    REPORT_OUT.write_text("\n".join(report), "utf-8")
    print(parser_table.to_string(index=False))
    print(main_table.to_string(index=False))


if __name__ == "__main__":
    main()
