"""Build the Exp28 train-only secondary-teacher review route."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INVENTORY_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28b_paper_train_annotation_inventory_seed42"
)
DEFAULT_REFERENCE = DEFAULT_INVENTORY_DIR / "private" / "exp28b_benchmark_reference_2654.jsonl"
DEFAULT_PACKETS = DEFAULT_INVENTORY_DIR / "private" / "exp28b_blind_teacher_packets_2654.jsonl"
DEFAULT_PROTOCOL_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28d_selective_secondary_route_seed42"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_fraction(sample_id: str) -> float:
    value = int(hashlib.sha256(f"exp28-high-control|{sample_id}".encode("utf-8")).hexdigest()[:12], 16)
    return value / float(16**12 - 1)


def selected_protocol(decision_path: Path, override: str | None) -> str:
    if override:
        return override
    if not decision_path.exists():
        raise FileNotFoundError(f"Protocol decision is missing: {decision_path}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    protocol = decision.get("selected_protocol")
    if decision.get("status") != "READY_FOR_SEALED_QUALIFICATION" or not protocol:
        raise ValueError("Protocol development has not selected a protocol")
    return str(protocol)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    protocol = selected_protocol(args.protocol_decision, args.protocol)
    primary_path = args.primary_dir / "qwen" / protocol / "all_train.jsonl"
    for path in (args.reference, args.packets, primary_path):
        if not path.exists():
            raise FileNotFoundError(path)
    references = {str(row["sample_id"]): row for row in read_jsonl(args.reference)}
    packets = {str(row["sample_id"]): row for row in read_jsonl(args.packets)}
    primary = {
        str(row["sample_id"]): row
        for row in read_jsonl(primary_path)
        if isinstance(row.get("annotation"), dict) and not row.get("schema_errors")
    }
    if len(references) != 2654 or len(packets) != 2654:
        raise ValueError("Exp28D requires the locked 2,654-row paper-train universe")
    if set(primary) != set(references):
        raise ValueError(f"Primary coverage mismatch: {len(primary)}/2654")

    routed_packets = []
    route_rows = []
    reason_counts = Counter()
    for sid, reference in references.items():
        annotation = primary[sid]["annotation"]
        original = int(reference["original_label"])
        teacher = int(annotation["score"])
        failures = list(annotation.get("major_failures") or [])
        reasons = []
        if original <= 2:
            reasons.append("original_low_score")
        if teacher != original:
            reasons.append("primary_original_disagreement")
        if abs(teacher - original) >= 2:
            reasons.append("large_score_gap")
        if float(annotation.get("confidence", 0.0)) < args.confidence_threshold:
            reasons.append("low_primary_confidence")
        if failures and failures != ["no_major_failure"]:
            reasons.append("primary_detected_failure")
        if annotation.get("score_cap") is not None:
            reasons.append("primary_score_cap")
        if original >= 4 and teacher == original and stable_fraction(sid) < args.high_control_rate:
            reasons.append("locked_high_score_control")
        route = bool(reasons)
        if route:
            packet = dict(packets[sid])
            packet["protocol_subset"] = "secondary_route"
            packet["route_reasons"] = reasons
            routed_packets.append(packet)
            reason_counts.update(reasons)
        route_rows.append(
            {
                "sample_id": sid,
                "original_label": original,
                "primary_score": teacher,
                "primary_confidence": annotation.get("confidence"),
                "routed": route,
                "route_reasons": "|".join(reasons),
                "reference_status": "teacher_routing_metadata_not_human_review",
            }
        )

    out = args.out_dir
    write_jsonl(out / "private" / "exp28d_secondary_teacher_packets.jsonl", routed_packets)
    write_csv(
        out / "tables" / "exp28d_secondary_route_manifest_light.csv",
        route_rows,
        ["sample_id", "original_label", "primary_score", "primary_confidence", "routed", "route_reasons", "reference_status"],
    )
    write_csv(
        out / "tables" / "exp28d_secondary_route_reason_counts.csv",
        [{"route_reason": reason, "count": count} for reason, count in reason_counts.most_common()],
        ["route_reason", "count"],
    )
    decision = {
        "status": "READY_FOR_SECONDARY_TEACHER",
        "selected_protocol": protocol,
        "paper_train_rows": len(references),
        "primary_rows": len(primary),
        "secondary_route_rows": len(routed_packets),
        "secondary_route_rate": len(routed_packets) / len(references),
        "confidence_threshold": args.confidence_threshold,
        "high_control_rate": args.high_control_rate,
        "dev_or_test_read": False,
        "next_action": "Run the selected protocol with DeepSeek on exp28d_secondary_teacher_packets.jsonl.",
    }
    write_json(out / "decision" / "exp28d_secondary_route_decision.json", decision)
    report = f"""# Exp28D Selective Secondary-Teacher Route

- selected protocol: `{protocol}`
- paper-train rows: {len(references)}
- primary Qwen rows: {len(primary)}
- routed to DeepSeek: {len(routed_packets)} ({len(routed_packets) / len(references):.2%})
- confidence threshold: {args.confidence_threshold}
- locked high-control rate: {args.high_control_rate:.0%}
- dev/test read: no

Routing is triggered by original low scores, primary/original disagreement, large score gaps,
low primary confidence, detected failures, score caps, and a deterministic sample of confirmed
high-score controls. The secondary teacher receives the same blind input and never sees the
original label or Qwen output.
"""
    report_path = out / "reports" / "exp28d_secondary_route_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--packets", type=Path, default=DEFAULT_PACKETS)
    parser.add_argument("--primary-dir", type=Path, default=DEFAULT_PROTOCOL_DIR / "private")
    parser.add_argument(
        "--protocol-decision",
        type=Path,
        default=DEFAULT_PROTOCOL_DIR / "decision" / "exp28c_protocol_development_protocol_decision.json",
    )
    parser.add_argument("--protocol", choices=["p0_holistic_zero_shot", "p1_rubric_first", "p2_rubric_verify_then_score"], default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--high-control-rate", type=float, default=0.10)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(prepare(parse_args()), ensure_ascii=False, sort_keys=True))
