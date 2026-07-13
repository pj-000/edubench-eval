"""Audit public aggregate novelty statistics for the selected Exp39B-R1 sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp39b_educfa_rlcr.common import R1_ROOT, read_csv, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=R1_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.out_dir / "tables/exp39b_r1_source_novelty_audit.csv")
    if len(rows) != 1:
        raise ValueError("Expected one aggregate source-novelty row")
    row = rows[0]
    checks = {
        "selected_rows_60": int(row["selected_rows"]) == 60,
        "selected_qkeys_60": int(row["selected_question_keys"]) == 60,
        "source_id_overlap_zero": int(row["source_id_overlap"]) == 0,
        "answer_hash_overlap_zero": int(row["exact_answer_hash_overlap"]) == 0,
        "assessment_key_overlap_zero": int(row["assessment_key_overlap"]) == 0,
        "char_similarity_below_threshold": float(row["max_char_5gram_similarity"]) < 0.90,
        "token_similarity_below_threshold": float(row["max_token_jaccard"]) < 0.90,
        "declared_scope": row["inference_scope"] == "response_disjoint_within_seen_question_clusters",
        "dev_test_zero": int(row["dev_access_count"]) == int(row["test_access_count"]) == 0,
    }
    decision = {"status": "NOVELTY_AUDIT_PASS" if all(checks.values()) else "NOVELTY_AUDIT_FAIL", "checks": checks}
    write_json(args.out_dir / "decision/exp39b_r1_source_novelty_decision.json", decision)
    print(json.dumps(decision, sort_keys=True))
    if decision["status"] != "NOVELTY_AUDIT_PASS":
        raise SystemExit(5)


if __name__ == "__main__":
    main()
