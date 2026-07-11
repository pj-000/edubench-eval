"""Validate Exp27Q Safe16 equivalence and dev-only isolation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.exp27p.common import input_source_hash, read_jsonl, stable_hash, write_csv
from thesis_exp.src.edujudge.exp27q import EXP27O_DIR, OUTPUT_DIR, SAFE16_DATASET


V3 = EXP27O_DIR / "private/data/exp27o_v3_selective_soft_audit_train.jsonl"
MANIFEST = EXP27O_DIR / "tables/exp27o_high_impact16_manifest_light.csv"
DEV = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
INHERITED_AUDIT = EXP27O_DIR / "tables/exp27o_leakage_audit.csv"


def validate(out_dir: Path, dataset: Path, dev_path: Path) -> dict[str, object]:
    v3 = read_jsonl(V3)
    safe = read_jsonl(dataset)
    dev = read_jsonl(dev_path)
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        manifest_ids = {row["sample_id"] for row in csv.DictReader(handle)}
    inherited = {}
    with INHERITED_AUDIT.open("r", encoding="utf-8", newline="") as handle:
        inherited = {row["check"]: int(row["count"]) for row in csv.DictReader(handle)}
    if any(inherited.get(key, -1) != 0 for key in ("train_question_overlap_test", "train_sample_overlap_test", "test_label_used") if key in inherited):
        raise ValueError("Inherited Exp27O test-isolation audit is not clean")
    safe_ids = [str(row["sample_id"]) for row in safe]
    changed = {a["sample_id"] for a, b in zip(v3, safe) if stable_hash(a) != stable_hash(b)}
    dev_ids = {str(row.get("sample_id") or row.get("record_id")) for row in dev}
    dev_keys = {str(row["question_key"]) for row in dev}
    train_keys = {str(row["question_key"]) for row in safe}
    checks = {
        "rows": len(safe),
        "unique_sample_id": len(set(safe_ids)),
        "changed_rows_vs_v3": len(changed),
        "changed_input_hashes": sum(input_source_hash(a) != input_source_hash(b) for a, b in zip(v3, safe)),
        "changed_sample_weights": sum(float(a["sample_weight"]) != float(b["sample_weight"]) for a, b in zip(v3, safe)),
        "non_safe16_target_mismatches": sum(
            (a["label_5"], a["soft_target_5"]) != (b["label_5"], b["soft_target_5"])
            for a, b in zip(v3, safe) if a["sample_id"] not in manifest_ids
        ),
        "safe16_id_symmetric_difference": len(changed ^ manifest_ids),
        "dev_sample_overlap": len(set(safe_ids) & dev_ids),
        "dev_question_overlap": len(train_keys & dev_keys),
        "test_sample_overlap_inherited": inherited.get("train_sample_overlap_test", inherited.get("audited_sample_overlap_test", 0)),
        "test_question_overlap_inherited": inherited.get("train_question_overlap_test", 0),
        "test_file_read": False,
        "test_label_read": False,
    }
    expected = {"rows": 3326, "unique_sample_id": 3326, "changed_rows_vs_v3": 16}
    passed = all(checks[key] == value for key, value in expected.items()) and all(
        checks[key] == 0 for key in (
            "changed_input_hashes", "changed_sample_weights", "non_safe16_target_mismatches",
            "safe16_id_symmetric_difference", "dev_sample_overlap", "dev_question_overlap",
            "test_sample_overlap_inherited", "test_question_overlap_inherited",
        )
    )
    checks["status"] = "PASS" if passed else "FAIL"
    write_csv(out_dir / "tables/exp27q_safe16_leakage_audit.csv", [{"check": key, "value": value} for key, value in checks.items()])
    if not passed:
        raise ValueError(json.dumps(checks, sort_keys=True))
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dataset", type=Path, default=SAFE16_DATASET)
    parser.add_argument("--dev", type=Path, default=DEV)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(validate(args.out_dir, args.dataset, args.dev), sort_keys=True))
