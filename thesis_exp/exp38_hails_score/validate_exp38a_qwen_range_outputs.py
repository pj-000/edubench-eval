"""Strictly validate Exp38A parsed Qwen score ranges."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import jsonschema
except ModuleNotFoundError:
    from thesis_exp.exp17_low_score_evidence import json_schema_compat as jsonschema

from thesis_exp.exp38_hails_score.common import ROOT, TRAIN_PATH, read_jsonl, sample_id, write_csv, write_json

SCHEMA_PATH = Path("thesis_exp/exp38_hails_score/schemas/exp38a_qwen_score_range_schema.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--split", choices=("qualification", "all_train"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.out_dir / f"parsed_ranges_private/exp38a_qwen_ranges_{args.split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Parsed Qwen ranges not found: {path}")
    rows = read_jsonl(path)
    expected = 196 if args.split == "qualification" else 2654
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    issues = []
    for row in rows:
        sid = sample_id(row)
        for error in validator.iter_errors(row):
            issues.append({"sample_id_hash": sid[:12], "issue": "schema", "detail": error.message})
        values = [row.get("minimum_plausible_score"), row.get("most_plausible_score"), row.get("maximum_plausible_score")]
        if all(isinstance(value, int) for value in values) and not (values[0] <= values[1] <= values[2]):
            issues.append({"sample_id_hash": sid[:12], "issue": "range_order", "detail": str(values)})
    duplicate_count = len(rows) - len({sample_id(row) for row in rows})
    valid = len(rows) == expected and duplicate_count == 0 and not issues and all(row.get("target_scope_confirmed") is True for row in rows)
    summary = {
        "split": args.split,
        "expected_rows": expected,
        "observed_rows": len(rows),
        "unique_sample_ids": len({sample_id(row) for row in rows}),
        "duplicate_count": duplicate_count,
        "schema_issue_count": len(issues),
        "schema_success_rate": 1.0 - len({row["sample_id_hash"] for row in issues}) / max(len(rows), 1),
        "target_scope_success_rate": sum(row.get("target_scope_confirmed") is True for row in rows) / max(len(rows), 1),
        "valid": valid,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    write_csv(args.out_dir / f"tables/exp38a_{args.split}_range_validation_issues.csv", issues)
    write_json(args.out_dir / f"decision/exp38a_{args.split}_range_validation.json", summary)
    if not valid:
        raise RuntimeError(json.dumps(summary, sort_keys=True))
    if args.split == "all_train":
        train = {sample_id(row): row for row in read_jsonl(TRAIN_PATH)}
        if set(train) != {sample_id(row) for row in rows}:
            raise RuntimeError("All-train ranges do not exactly match paper-like train IDs")
        widths = Counter()
        transitions = Counter()
        by_label: dict[int, list[int]] = defaultdict(list)
        for row in rows:
            sid = sample_id(row)
            source = train[sid]
            minimum = int(row["minimum_plausible_score"])
            center = int(row["most_plausible_score"])
            maximum = int(row["maximum_plausible_score"])
            width = maximum - minimum + 1
            label = int(source["label_5"])
            widths[(minimum, center, maximum, width, row["range_confidence"])] += 1
            transitions[(label, center)] += 1
            by_label[label].append(width)
        write_csv(args.out_dir / "tables/exp38a_full_train_range_distribution.csv", [
            {"minimum": key[0], "center": key[1], "maximum": key[2], "width": key[3], "confidence": key[4], "count": count}
            for key, count in sorted(widths.items())
        ])
        write_csv(args.out_dir / "tables/exp38a_full_train_direction_transitions.csv", [
            {"human_label": key[0], "qwen_center": key[1], "count": count}
            for key, count in sorted(transitions.items())
        ])
        write_csv(args.out_dir / "tables/exp38a_range_width_by_human_label.csv", [
            {"human_label": label, "n": len(values), "mean_width": sum(values) / len(values), "non_singleton_rate": sum(value > 1 for value in values) / len(values)}
            for label, values in sorted(by_label.items())
        ])
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
