"""Static, reproducible audit of the current Exp27I calibration implementation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.utils.io import write_csv, write_json  # noqa: E402


DEFAULT_SOURCE = Path("thesis_exp/exp17_low_score_evidence/build_exp27i_codex_calibrated_dataset.py")
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27j_independent_audit_seed42"
)


def audit(source: Path, out_dir: Path) -> dict[str, Any]:
    text = source.read_text(encoding="utf-8")
    independent_patterns = [
        r"adjudication[_-]input",
        r"adjudication[_-]filled",
        r"independent[_-]adjudication",
    ]
    independent_input = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in independent_patterns)
    queue_slice = bool(re.search(r"queue\s*\[:\s*80\s*\]", text))
    top80_written = "exp27i_codex_top80_direct_review.csv" in text and "write_csv" in text
    reliability_constants = all(token in text for token in ["score += 2.0", "score += 1.0", "score -= 0.75"])
    individual_human_used = any(token in text for token in ["human_1_5", "human_2_5", "human_3_5"])
    score_derived_risk = '"hidden_low_failure" if row["calibrated_score"] <= 2' in text
    evidence_fields = ["evidence_span", "rubric_clause", "failure_visibility"]
    assistant_block = text[text.find("assistant = {") : text.find("user =", text.find("assistant = {"))]
    preserved = all(field in assistant_block for field in evidence_fields)

    checks = [
        (
            "independent_top80_adjudication_input_found",
            independent_input,
            "The builder must consume an external filled adjudication artifact before claiming direct review.",
        ),
        (
            "top80_defined_by_queue_slice",
            queue_slice,
            "Current top-80 membership is derived from queue[:80].",
        ),
        (
            "top80_review_csv_is_generated_output",
            top80_written,
            "The named direct-review CSV is written by the builder rather than read as an input.",
        ),
        (
            "teacher_reliability_uses_fixed_constants",
            reliability_constants,
            "Reliability increments are fixed heuristics rather than estimates from an independent audit set.",
        ),
        (
            "human_individual_scores_used",
            individual_human_used,
            "Individual human scores should inform uncertainty and review priority.",
        ),
        (
            "risk_flag_is_score_derived",
            score_derived_risk,
            "The exported risk flag is mechanically derived from calibrated_score <= 2.",
        ),
        (
            "evidence_fields_preserved_in_sft",
            preserved,
            "Evidence span, rubric clause, and failure visibility should be retained if used as supervision.",
        ),
    ]
    write_csv(
        out_dir / "tables" / "exp27j_exp27i_implementation_audit.csv",
        [
            {"check": name, "value": str(value).lower(), "interpretation": interpretation}
            for name, value, interpretation in checks
        ],
    )
    current_rule_based = queue_slice and top80_written and not independent_input
    decision = {
        "independent_top80_adjudication_input_found": independent_input,
        "current_codex_top80_is_rule_based": current_rule_based,
        "top80_review_csv_is_generated_output": top80_written,
        "teacher_reliability_uses_fixed_constants": reliability_constants,
        "human_individual_scores_used": individual_human_used,
        "risk_flag_is_score_derived": score_derived_risk,
        "evidence_fields_preserved_in_sft": preserved,
        "exp27i_current_292_formal_training_ready": False if current_rule_based else None,
        "recommendation": (
            "revise_calibration_after_independent_audit"
            if current_rule_based
            else "validate_calibration_against_independent_adjudication"
        ),
    }
    write_json(out_dir / "decision" / "exp27j_exp27i_implementation_audit.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Exp27I calibration implementation.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.out_dir), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
