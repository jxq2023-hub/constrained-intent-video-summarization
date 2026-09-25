from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    requests = ROOT / "data/query_requests/language_requests_1080.jsonl"
    request_count = sum(1 for line in requests.read_text("utf-8").splitlines() if line.strip())
    assert request_count == 1080, request_count

    parser_rows = rows(ROOT / "data/intent_outputs/language_request_parser_outputs.csv")
    assert len(parser_rows) == 4320, len(parser_rows)

    method_rows = rows(ROOT / "data/selected_shots/qfvs_method_outputs.csv")
    assert len(method_rows) == 1800, len(method_rows)
    for row in method_rows:
        value = json.loads(row["selected_indices"])
        assert isinstance(value, list) and all(isinstance(item, int) for item in value)

    human_rows = rows(ROOT / "data/human_evaluation/human_ratings_long.csv")
    assert len(human_rows) == 400, len(human_rows)
    raters = sorted({row["rater_id"] for row in human_rows})
    assert raters == ["R001", "R002", "R003", "R004", "R005"], raters

    forbidden_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".pth", ".pt", ".ckpt", ".h5", ".hdf5", ".zip", ".rar", ".7z"}
    forbidden_names = {"private_blinding_key.csv"}
    for path in ROOT.rglob("*"):
        if path.is_file():
            assert path.suffix.lower() not in forbidden_suffixes, path
            assert path.name.lower() not in forbidden_names, path

    print("Release verification passed")
    print(f"requests={request_count}, parser_rows={len(parser_rows)}, method_rows={len(method_rows)}, human_ratings={len(human_rows)}")


if __name__ == "__main__":
    main()
