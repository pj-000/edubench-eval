"""Validate Exp24 R7G ORC-DPO metadata before training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import clean, clamp_score  # noqa: E402
from thesis_exp.src.edujudge.utils.io import write_json, write_text  # noqa: E402


DEFAULT_DATA = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp24_orc_dpo_seed42/"
    "data/edubench_r7g_orc_score_channel_reason_aux_train.json"
)
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp24_orc_dpo_seed42")


REQUIRED_FIELDS = {
    "messages",
    "chosen_score_response",
    "rejected_score_response",
    "auxiliary_reason_target",
    "pair_id",
    "source_sample_id",
    "question_key",
    "gold_label",
    "rejected_score",
    "risk_type",
    "score_derived_risk_type",
    "risk_type_matches_score_direction",
    "ordinal_distance",
    "has_human_reason",
    "reason_hash",
    "LH",
    "LM",
    "HL",
    "HM",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"R7G data must be a list: {path}")
    return data


def parse_payload(message: dict[str, Any]) -> dict[str, Any]:
    try:
        data = json.loads(clean(message.get("content")))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def risk_type_from_scores(gold: int, rejected: int) -> str:
    if gold <= 2 and rejected >= 4:
        return "low_to_high_real_model_error"
    if gold <= 2 and rejected == 3:
        return "low_to_mid_real_model_error"
    if gold >= 4 and rejected <= 2:
        return "high_to_low_real_model_error"
    if gold >= 4 and rejected == 3:
        return "high_to_mid_real_model_error"
    if rejected > gold:
        return "upward_real_model_error"
    if rejected < gold:
        return "downward_real_model_error"
    return "same_score_non_error"


def validate(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(args.data)
    errors: list[str] = []
    pair_ids: set[str] = set()
    risk_counts: Counter[str] = Counter()
    distance_counts: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            errors.append(f"row {idx} missing fields: {missing}")
            continue
        pair_id = clean(row.get("pair_id"))
        if not pair_id:
            errors.append(f"row {idx} empty pair_id")
        if pair_id in pair_ids:
            errors.append(f"row {idx} duplicated pair_id: {pair_id}")
        pair_ids.add(pair_id)

        gold = clamp_score(row.get("gold_label"))
        rejected = clamp_score(row.get("rejected_score"))
        chosen = parse_payload(row["chosen_score_response"])
        bad = parse_payload(row["rejected_score_response"])
        if chosen != {"score": gold}:
            errors.append(f"row {idx} chosen_score_response mismatch: {chosen} != score {gold}")
        if bad != {"score": rejected}:
            errors.append(f"row {idx} rejected_score_response mismatch: {bad} != score {rejected}")
        if int(row.get("ordinal_distance", -1)) != abs(gold - rejected):
            errors.append(f"row {idx} ordinal_distance mismatch")
        expected_flags = {
            "LH": int(gold <= 2 and rejected >= 4),
            "LM": int(gold <= 2 and rejected == 3),
            "HL": int(gold >= 4 and rejected <= 2),
            "HM": int(gold >= 4 and rejected == 3),
        }
        reason = clean(row.get("auxiliary_reason_target"))
        if not reason:
            errors.append(f"row {idx} missing auxiliary_reason_target")
        prompt_text = json.dumps(row.get("messages") or [], ensure_ascii=False)
        if reason and reason in prompt_text:
            errors.append(f"row {idx} human reason appears in prompt")
        if gold == rejected:
            errors.append(f"row {idx} gold_label equals rejected_score; not a preference pair")
        for flag, expected in expected_flags.items():
            actual = int(row.get(flag, 0))
            if actual not in {0, 1}:
                errors.append(f"row {idx} invalid {flag}: {row.get(flag)}")
            if actual != expected:
                errors.append(f"row {idx} {flag} mismatch: {actual} != {expected}")
        expected_risk_type = risk_type_from_scores(gold, rejected)
        if clean(row.get("score_derived_risk_type")) != expected_risk_type:
            errors.append(
                f"row {idx} score_derived_risk_type mismatch: "
                f"{row.get('score_derived_risk_type')} != {expected_risk_type}"
            )
        if clean(row.get("risk_type")) != expected_risk_type:
            errors.append(f"row {idx} risk_type mismatch: {row.get('risk_type')} != {expected_risk_type}")
        if row.get("risk_type_matches_score_direction") is not True:
            errors.append(f"row {idx} risk_type_matches_score_direction is not true")
        risk_counts[clean(row.get("risk_type"))] += 1
        distance_counts[str(row.get("ordinal_distance"))] += 1

    summary = {
        "data": str(args.data),
        "rows": len(rows),
        "unique_pair_ids": len(pair_ids),
        "errors": len(errors),
        "passed": not errors,
        "risk_type_counts": dict(sorted(risk_counts.items())),
        "ordinal_distance_counts": dict(sorted(distance_counts.items())),
        "sample_errors": errors[:20],
    }
    write_json(args.out_dir / "decision" / "exp24_orc_metadata_validation.json", summary)
    lines = [
        "# Exp24 ORC Metadata Validation",
        "",
        f"- data: `{args.data}`",
        f"- rows: {len(rows)}",
        f"- unique_pair_ids: {len(pair_ids)}",
        f"- errors: {len(errors)}",
        f"- passed: {not errors}",
    ]
    if errors:
        lines.extend(["", "## Sample Errors", ""])
        lines.extend(f"- {error}" for error in errors[:20])
    write_text(args.out_dir / "reports" / "exp24_orc_metadata_validation.md", "\n".join(lines))
    if errors:
        raise SystemExit(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Exp24 R7G ORC-DPO metadata.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(validate(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
