from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def mean_by(rows, group, value):
    grouped = defaultdict(list)
    for row in rows:
        try:
            grouped[row[group]].append(float(row[value]))
        except (KeyError, TypeError, ValueError):
            pass
    return {key: sum(vals) / len(vals) for key, vals in sorted(grouped.items()) if vals}


def main():
    qfvs = read(ROOT / "data/selected_shots/qfvs_method_outputs.csv")
    human = read(ROOT / "data/human_evaluation/human_ratings_long.csv")
    print("QFVS mean semantic F-score by method")
    for key, value in mean_by(qfvs, "method", "fscore").items():
        print(f"  {key}: {value:.6f}")
    print("Human mean overall usefulness by method")
    for key, value in mean_by(human, "method", "overall_usefulness_1_5").items():
        print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    main()
