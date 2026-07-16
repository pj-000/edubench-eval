"""Validate row-level completeness and blindness compliance of verifier output."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import OUT, PRIVATE, VERIFIER_STATES, ensure_output_layout, read_jsonl, write_csv, write_json

try:
    import jsonschema
except ImportError as exc:  # pragma: no cover - environment preflight
    raise RuntimeError("Exp48A verifier validation requires jsonschema") from exc


def validate_one(packet_rows: list[dict], output_rows: list[dict], verifier: str, schema: dict) -> tuple[list[dict], dict]:
    packets = {str(row["family_id"]): row for row in packet_rows}
    outputs = {str(row.get("family_id", "")): row for row in output_rows}
    errors = []
    complete = 0
    model_families = set()
    sessions = set()
    for family_id, packet in packets.items():
        output = outputs.get(family_id)
        row_errors = []
        if output is None:
            row_errors.append("missing_family_output")
        else:
            try:
                jsonschema.validate(output, schema)
            except jsonschema.ValidationError as exc:
                row_errors.append(f"json_schema_invalid:{exc.json_path}")
            provenance = output.get("verifier_provenance", {})
            model_families.add(str(provenance.get("model_family", "unknown")))
            sessions.add(str(provenance.get("session_id", "unknown")))
            if "score" in output or "intended_score" in output:
                row_errors.append("direct_score_field_forbidden")
            expected_answers = {row["anonymous_answer_id"] for row in packet["answers"]}
            expected_criteria = {row["id"] for row in packet["criteria"]}
            observed_answers = {row.get("anonymous_answer_id") for row in output.get("answers", [])}
            if observed_answers != expected_answers:
                row_errors.append("answer_coverage_mismatch")
            for answer in output.get("answers", []):
                if answer.get("uncertainty") not in {"low", "medium", "high"}:
                    row_errors.append("uncertainty_invalid")
                observed_criteria = {row.get("criterion_id") for row in answer.get("criteria", [])}
                if observed_criteria != expected_criteria:
                    row_errors.append("criterion_coverage_mismatch")
                for item in answer.get("criteria", []):
                    if item.get("status") not in VERIFIER_STATES:
                        row_errors.append("criterion_status_invalid")
                    if not str(item.get("evidence_span", "")).strip() and not str(item.get("missing_reason", "")).strip():
                        row_errors.append("evidence_or_missing_reason_required")
        if not row_errors:
            complete += 1
        else:
            errors.append({"verifier": verifier, "family_id": family_id, "errors": sorted(set(row_errors))})
    summary = {
        "verifier": verifier, "packet_families": len(packets), "output_families": len(output_rows),
        "complete_families": complete, "invalid_families": len(packets) - complete,
        "model_families": sorted(model_families), "session_ids": sorted(sessions),
    }
    return errors, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier-a", type=Path, default=PRIVATE / "verifier_a/exp48a_verifier_a_outputs.jsonl")
    parser.add_argument("--verifier-b", type=Path, default=PRIVATE / "verifier_b/exp48a_verifier_b_outputs.jsonl")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    ensure_output_layout(args.out_dir)
    summaries, all_errors = [], []
    schema = json.loads((Path(__file__).resolve().parent / "schemas/exp48a_criterion_verification_schema.json").read_text(encoding="utf-8"))
    for verifier, output_path in (("a", args.verifier_a), ("b", args.verifier_b)):
        packets = read_jsonl(args.out_dir / f"private/verifier_packets/exp48a_verifier_{verifier}_packets.jsonl")
        outputs = read_jsonl(output_path)
        errors, summary = validate_one(packets, outputs, verifier, schema)
        summaries.append(summary)
        all_errors.extend(errors)
    write_csv(args.out_dir / "tables/exp48a_verifier_completion.csv", [{**row, "model_families": "|".join(row["model_families"]), "session_ids": "|".join(row["session_ids"])} for row in summaries])
    write_json(args.out_dir / "private/adjudication/exp48a_verifier_validation_errors.json", all_errors)
    decision = {"status": "VERIFIER_OUTPUTS_VALID" if not all_errors else "VERIFIER_OUTPUTS_INVALID", "verifiers": summaries, "error_families": len(all_errors)}
    write_json(args.out_dir / "private/adjudication/exp48a_verifier_validation_decision.json", decision)
    print(json.dumps(decision, sort_keys=True))
    if all_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
