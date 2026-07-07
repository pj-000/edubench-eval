"""Validate Exp25 Structured SRC-DPO metadata and guardrails."""

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

from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import clean  # noqa: E402
from thesis_exp.exp17_low_score_evidence.prepare_exp25_structured_src_dpo import split_ids  # noqa: E402
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_DATA = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp25_structured_src_dpo_seed42/"
    "data/edubench_r7h_structured_src_dpo_train.json"
)
DEFAULT_DEV_JSONL = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_TEST_JSONL = Path("thesis_exp/data/splits/question_seed42/test.jsonl")
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp25_structured_src_dpo_seed42")

REQUIRED_FIELDS = [
    "pair_id",
    "source_sample_id",
    "question_key",
    "gold_label",
    "rejected_score",
    "risk_type",
    "negative_type",
    "negative_source",
    "pair_weight",
    "ordinal_distance",
    "has_human_reason",
    "reason_hash",
    "reason_in_prompt",
    "train_only",
]


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    return data


def parse_content(message: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(clean(message.get("content")))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def prompt_text(row: dict[str, Any]) -> str:
    return "\n".join(clean(message.get("content")) for message in row.get("messages") or [])


def schema_keys(message: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(parse_content(message).keys()))


def validate(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_json(args.data)
    dev_ids, dev_qkeys, dev_n = split_ids(args.dev_jsonl)
    test_ids, test_qkeys, test_n = split_ids(args.test_jsonl)
    errors: list[dict[str, Any]] = []
    negative_counts = Counter()
    weight_counts = Counter()
    source_ids: set[str] = set()
    question_keys: set[str] = set()
    pair_ids: set[str] = set()
    counterfactual_count = 0
    same_schema_count = 0
    chosen_reason_prompt_count = 0
    rejected_reason_prompt_count = 0

    for idx, row in enumerate(rows):
        row_errors: list[str] = []
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            row_errors.append("missing_fields:" + "|".join(missing))
        pair_id = clean(row.get("pair_id"))
        if not pair_id:
            row_errors.append("missing_pair_id")
        elif pair_id in pair_ids:
            row_errors.append("duplicate_pair_id")
        pair_ids.add(pair_id)
        sid = clean(row.get("source_sample_id"))
        qkey = clean(row.get("question_key"))
        if sid:
            source_ids.add(sid)
        if qkey:
            question_keys.add(qkey)
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not isinstance(chosen, dict) or not isinstance(rejected, dict):
            row_errors.append("chosen_rejected_not_message_dict")
        else:
            chosen_payload = parse_content(chosen)
            rejected_payload = parse_content(rejected)
            if not chosen_payload or not rejected_payload:
                row_errors.append("chosen_rejected_not_valid_json")
            if schema_keys(chosen) != schema_keys(rejected):
                row_errors.append("chosen_rejected_schema_mismatch")
            else:
                same_schema_count += 1
            reason = clean(chosen_payload.get("reason"))
            prompt = prompt_text(row)
            if reason and reason[:80] in prompt:
                chosen_reason_prompt_count += 1
                row_errors.append("chosen_reason_found_in_prompt")
            rejected_reason = clean(rejected_payload.get("reason"))
            if rejected_reason and rejected_reason[:80] in prompt:
                rejected_reason_prompt_count += 1
                row_errors.append("rejected_reason_found_in_prompt")
        negative_type = clean(row.get("negative_type"))
        negative_source = clean(row.get("negative_source"))
        negative_counts[negative_type] += 1
        weight_counts[str(row.get("pair_weight"))] += 1
        if negative_type == "low_failure_erasure_counterfactual":
            counterfactual_count += 1
            if negative_source != "counterfactual_failure_erasure_train_only_not_human":
                row_errors.append("bad_counterfactual_negative_source")
        if row.get("reason_in_prompt") not in {False, "False", "false", 0, "0"}:
            row_errors.append("reason_in_prompt_not_false")
        if row.get("train_only") not in {True, "True", "true", 1, "1"}:
            row_errors.append("train_only_not_true")
        if row_errors:
            errors.append(
                {
                    "row_index": idx,
                    "pair_id": pair_id,
                    "source_sample_id": sid,
                    "negative_type": negative_type,
                    "errors": ";".join(row_errors),
                }
            )

    leakage_rows = [
        {
            "dataset": args.data.name,
            "rows": len(rows),
            "dev_rows_read_for_id_guard": dev_n,
            "test_rows_read_for_id_guard": test_n,
            "dev_sample_overlap": len(source_ids & dev_ids),
            "dev_question_overlap": len(question_keys & dev_qkeys),
            "test_sample_overlap": len(source_ids & test_ids),
            "test_question_overlap": len(question_keys & test_qkeys),
            "test_label_read": False,
        }
    ]
    validation_rows = [
        {"metric": "row_count", "value": len(rows)},
        {"metric": "unique_pair_id_count", "value": len(pair_ids)},
        {"metric": "error_count", "value": len(errors)},
        {"metric": "same_schema_pair_count", "value": same_schema_count},
        {"metric": "counterfactual_count", "value": counterfactual_count},
        {"metric": "chosen_reason_found_in_prompt_count", "value": chosen_reason_prompt_count},
        {"metric": "rejected_reason_found_in_prompt_count", "value": rejected_reason_prompt_count},
        {"metric": "dev_sample_overlap", "value": len(source_ids & dev_ids)},
        {"metric": "dev_question_overlap", "value": len(question_keys & dev_qkeys)},
        {"metric": "test_sample_overlap", "value": len(source_ids & test_ids)},
        {"metric": "test_question_overlap", "value": len(question_keys & test_qkeys)},
        {"metric": "test_label_read", "value": False},
    ]
    for name, count in sorted(negative_counts.items()):
        validation_rows.append({"metric": f"negative_type:{name}", "value": count})
    for weight, count in sorted(weight_counts.items()):
        validation_rows.append({"metric": f"pair_weight:{weight}", "value": count})

    passed = (
        not errors
        and len(source_ids & dev_ids) == 0
        and len(question_keys & dev_qkeys) == 0
        and len(source_ids & test_ids) == 0
        and len(question_keys & test_qkeys) == 0
    )
    decision = {
        "data": str(args.data),
        "passed": passed,
        "error_count": len(errors),
        "rows": len(rows),
        "same_schema_pair_count": same_schema_count,
        "negative_type_counts": dict(sorted(negative_counts.items())),
        "test_label_read": False,
        "chosen_reason_in_prompt": chosen_reason_prompt_count > 0,
        "rejected_reason_in_prompt": rejected_reason_prompt_count > 0,
        "human_reason_in_prompt": chosen_reason_prompt_count > 0 or rejected_reason_prompt_count > 0,
    }
    write_csv(args.out_dir / "tables" / "exp25_src_metadata_validation.csv", validation_rows)
    write_csv(args.out_dir / "tables" / "exp25_src_metadata_validation_errors.csv", errors)
    write_csv(args.out_dir / "tables" / "exp25_src_validation_leakage_audit.csv", leakage_rows)
    write_json(args.out_dir / "decision" / "exp25_src_metadata_validation_decision.json", decision)
    lines = [
        "# Exp25 SRC-DPO Metadata Validation",
        "",
        f"- data: `{args.data}`",
        f"- passed: `{passed}`",
        f"- rows: {len(rows)}",
        f"- errors: {len(errors)}",
        f"- same-schema pairs: {same_schema_count}",
        f"- chosen_reason_found_in_prompt_count: {chosen_reason_prompt_count}",
        f"- rejected_reason_found_in_prompt_count: {rejected_reason_prompt_count}",
        f"- test_label_read: `False`",
        "",
        "## Negative Types",
        "",
    ]
    for name, count in sorted(negative_counts.items()):
        lines.append(f"- {name}: {count}")
    write_text(args.out_dir / "reports" / "exp25_src_metadata_validation_report.md", "\n".join(lines))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Exp25 SRC-DPO metadata.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV_JSONL)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(validate(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
