"""Validate Exp48C outputs without consulting intended scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_json
from .exp48c_common import OUT, output_path, packet_path, validate_pointwise_output


def validate_verifier(verifier: str, packets_file: Path, outputs_file: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    packets = read_jsonl(packets_file)
    outputs = read_jsonl(outputs_file) if outputs_file.exists() else []
    packet_by_id = {row["packet_id"]: row for row in packets}
    output_by_id = {row.get("packet_id"): row for row in outputs}
    errors: list[dict[str, Any]] = []
    quote_valid = 0
    evidence_total = evidence_valid = 0
    score_range_valid = provenance_complete = 0
    for packet_id, packet in packet_by_id.items():
        output = output_by_id.get(packet_id)
        if output is None:
            errors.append({"packet_id": packet_id, "errors": ["missing_output"]})
            continue
        row_errors = validate_pointwise_output(output, packet)
        if not any("rubric_quote" in error for error in row_errors):
            quote_valid += 1
        if not any("score_range" in error or "score_outside" in error for error in row_errors):
            score_range_valid += 1
        if not any("provenance" in error for error in row_errors):
            provenance_complete += 1
        spans = output.get("answer_evidence_spans", [])
        if isinstance(spans, list):
            for span in spans:
                if span:
                    evidence_total += 1
                    evidence_valid += int(isinstance(span, str) and span in packet["answer"])
        if row_errors:
            errors.append({"packet_id": packet_id, "errors": row_errors})
    extra = sorted(set(output_by_id) - set(packet_by_id), key=str)
    for packet_id in extra:
        errors.append({"packet_id": str(packet_id), "errors": ["unexpected_output"]})
    summary = {
        "verifier": verifier, "packet_rows": len(packets), "output_rows": len(outputs),
        "unique_packet_ids": len(output_by_id), "valid_rows": len(packets) - sum(row["packet_id"] in packet_by_id for row in errors),
        "complete": len(packets) == 36 and len(outputs) == 36 and not errors,
        "schema_success_rate": (len(packets) - sum(row["packet_id"] in packet_by_id for row in errors)) / max(1, len(packets)),
        "rubric_quote_valid_count": quote_valid,
        "rubric_quote_validity": quote_valid / max(1, len(packets)),
        "evidence_total": evidence_total, "evidence_valid": evidence_valid,
        "evidence_validity": evidence_valid / max(1, evidence_total),
        "score_range_validity": score_range_valid / max(1, len(packets)),
        "provenance_completeness": provenance_complete / max(1, len(packets)),
        "error_rows": len(errors),
    }
    return summary, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier", choices=["codex", "qwen"], required=True)
    parser.add_argument("--packets", type=Path)
    parser.add_argument("--outputs", type=Path)
    parser.add_argument("--validation-json", type=Path)
    args = parser.parse_args()
    packets = args.packets or packet_path(args.verifier)
    outputs = args.outputs or output_path(args.verifier)
    summary, errors = validate_verifier(args.verifier, packets, outputs)
    target = args.validation_json or OUT / f"private/validation/exp48c_{args.verifier}_validation.json"
    write_json(target, {"summary": summary, "errors": errors})
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if outputs.exists() and not summary["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
