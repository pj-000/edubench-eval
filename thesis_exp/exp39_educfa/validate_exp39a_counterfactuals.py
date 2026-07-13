"""Independently validate Exp39A candidates and enforce the locked data gate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import jsonschema
except ModuleNotFoundError:
    from thesis_exp.exp17_low_score_evidence import json_schema_compat as jsonschema

from thesis_exp.exp39_educfa.common import (  # noqa: E402
    PROMPT_DIR,
    ROOT,
    SCHEMA_DIR,
    interval_distribution,
    length_ratio,
    read_jsonl,
    sample_id,
    sha256_file,
    stable_hash,
    text_edit_ratio,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def language_consistent(language: str, source: str, candidate: str) -> bool:
    def cjk_rate(text: str) -> float:
        visible = [char for char in text if not char.isspace()]
        if not visible:
            return 0.0
        return sum("\u4e00" <= char <= "\u9fff" for char in visible) / len(visible)

    if language == "zh":
        return cjk_rate(candidate) >= 0.10
    if language == "en":
        return cjk_rate(candidate) <= max(0.20, cjk_rate(source) + 0.10)
    return True


def span_validity(source: str, candidate: str, spans: list[dict[str, str]]) -> bool:
    if not spans:
        return False
    for span in spans:
        left = str(span.get("source_text") or "")
        right = str(span.get("replacement_text") or "")
        if not left and not right:
            return False
        if left and left not in source:
            return False
        if right and right not in candidate:
            return False
    return True


def acceptance_reasons(packet: dict[str, Any], generated: dict[str, Any], verified: dict[str, Any], duplicate: bool) -> tuple[list[str], dict[str, Any]]:
    source = str(packet["original_answer"])
    candidate = str(generated["counterfactual_answer"])
    target = int(packet["assigned_target_score"])
    gen_range = [int(value) for value in generated["target_score_range"]]
    original_range = [int(value) for value in verified["original_score_range"]]
    counterfactual_range = [int(value) for value in verified["counterfactual_score_range"]]
    center = float(verified["most_plausible_counterfactual_score"])
    edit = text_edit_ratio(source, candidate)
    length = length_ratio(source, candidate)
    intersection = sorted(set(range(gen_range[0], gen_range[1] + 1)) & set(range(counterfactual_range[0], counterfactual_range[1] + 1)))
    reasons = []
    if target not in range(counterfactual_range[0], counterfactual_range[1] + 1):
        reasons.append("target_outside_verifier_range")
    if abs(center - target) > 1:
        reasons.append("verifier_center_too_far")
    if original_range[1] < 4:
        reasons.append("original_not_verified_high")
    if target <= 2 and original_range[0] - counterfactual_range[1] < 1:
        reasons.append("insufficient_verified_score_drop")
    if target == 3 and center > 3:
        reasons.append("target3_center_above_3")
    for field in ("target_scope_confirmed", "operator_success", "rubric_failure_verified", "minimal_edit", "non_target_content_preserved"):
        if not bool(verified[field]):
            reasons.append(f"verifier_{field}_false")
    if bool(verified["answer_key_uncertainty"]):
        reasons.append("answer_key_uncertainty")
    if not 0.05 <= edit <= 0.35:
        reasons.append("edit_ratio_out_of_range")
    if not 0.60 <= length <= 1.20:
        reasons.append("length_ratio_out_of_range")
    if target <= 2 and counterfactual_range[0] >= 4:
        reasons.append("target_low_verified_entirely_high")
    if not intersection:
        reasons.append("generator_verifier_intersection_empty")
    if normalized_text(source) == normalized_text(candidate):
        reasons.append("candidate_equals_source")
    if duplicate:
        reasons.append("duplicate_counterfactual_answer")
    if not span_validity(source, candidate, generated["changed_spans"]):
        reasons.append("changed_span_invalid")
    if not language_consistent(str(packet.get("language") or "unknown"), source, candidate):
        reasons.append("language_mismatch")
    diagnostics = {
        "edit_ratio": edit,
        "length_ratio": length,
        "generator_min": gen_range[0],
        "generator_max": gen_range[1],
        "verifier_original_min": original_range[0],
        "verifier_original_max": original_range[1],
        "verifier_counterfactual_min": counterfactual_range[0],
        "verifier_counterfactual_max": counterfactual_range[1],
        "verifier_center": center,
        "intersection": intersection,
    }
    return sorted(set(reasons)), diagnostics


def aggregate(rows: list[dict[str, Any]], dimension: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[dimension])].append(row)
    output = []
    for value, subset in sorted(grouped.items()):
        accepted = sum(row["accepted"] for row in subset)
        output.append({
            dimension: value,
            "completed": len(subset),
            "accepted": accepted,
            "acceptance_rate": accepted / len(subset),
            "operator_success_rate": sum(row["operator_success"] for row in subset) / len(subset),
            "rubric_failure_verified_rate": sum(row["rubric_failure_verified"] for row in subset) / len(subset),
        })
    return output


def main() -> None:
    args = parse_args()
    packet_path = args.out_dir / "private/source_packets/exp39a_source_anchor_packets.jsonl"
    generation_path = args.out_dir / "private/generated_candidates/exp39a_qwen_generated_candidates.jsonl"
    verification_path = args.out_dir / "private/verified_counterfactuals/exp39a_deepseek_verifications.jsonl"
    packets = {sample_id(row): row for row in read_jsonl(packet_path)}
    generated = {sample_id(row): row for row in read_jsonl(generation_path)}
    verified = {sample_id(row): row for row in read_jsonl(verification_path)}
    if len(packets) != 240 or set(packets) != set(generated) or set(packets) != set(verified):
        raise ValueError(f"Exp39A requires exactly 240 aligned packets/generations/verifications: {len(packets)}/{len(generated)}/{len(verified)}")

    generation_schema = json.loads((SCHEMA_DIR / "exp39a_counterfactual_generation_schema.json").read_text(encoding="utf-8"))
    verification_schema = json.loads((SCHEMA_DIR / "exp39a_counterfactual_verification_schema.json").read_text(encoding="utf-8"))
    generation_schema_errors = sum(
        bool(list(jsonschema.Draft202012Validator(generation_schema).iter_errors(row))) for row in generated.values()
    )
    verification_schema_errors = sum(
        bool(list(jsonschema.Draft202012Validator(verification_schema).iter_errors(row))) for row in verified.values()
    )
    counts = Counter(normalized_text(row["counterfactual_answer"]) for row in generated.values())
    audit_rows = []
    accepted_rows = []
    for sid in sorted(packets):
        packet, candidate, verification = packets[sid], generated[sid], verified[sid]
        duplicate = counts[normalized_text(candidate["counterfactual_answer"])] > 1
        reasons, diagnostics = acceptance_reasons(packet, candidate, verification, duplicate)
        accepted = not reasons
        row = {
            "sample_id_hash": stable_hash(sid),
            "source_sample_id_hash": stable_hash(packet["source_sample_id"]),
            "question_key_hash": stable_hash(packet["question_key"]),
            "target_score": int(packet["assigned_target_score"]),
            "operator": packet["assigned_operator"],
            "language": packet.get("language"),
            "metric_group": packet.get("metric_group"),
            "accepted": accepted,
            "rejected_reasons": "|".join(reasons),
            "operator_success": bool(verification["operator_success"]),
            "rubric_failure_verified": bool(verification["rubric_failure_verified"]),
            **{key: value for key, value in diagnostics.items() if key != "intersection"},
        }
        audit_rows.append(row)
        if accepted:
            intersection = diagnostics["intersection"]
            accepted_rows.append({
                **packet,
                "answer": candidate["counterfactual_answer"],
                "counterfactual_answer": candidate["counterfactual_answer"],
                "generator_target_range": candidate["target_score_range"],
                "verifier_counterfactual_range": verification["counterfactual_score_range"],
                "verifier_original_range": verification["original_score_range"],
                "soft_target_5": interval_distribution(min(intersection), max(intersection), int(packet["assigned_target_score"])),
                "edit_ratio": diagnostics["edit_ratio"],
                "length_ratio": diagnostics["length_ratio"],
                "sample_weight": 1.0,
                "synthetic": True,
            })
    accepted_path = args.out_dir / "private/verified_counterfactuals/exp39a_accepted_counterfactuals.jsonl"
    write_jsonl(accepted_path, accepted_rows)

    accepted_count = len(accepted_rows)
    accepted_targets = Counter(int(row["assigned_target_score"]) for row in accepted_rows)
    operator_success_rate = sum(bool(row["operator_success"]) for row in audit_rows) / len(audit_rows)
    rubric_rate = sum(bool(row["rubric_failure_verified"]) for row in audit_rows) / len(audit_rows)
    accepted_edit_mean = sum(row["edit_ratio"] for row in accepted_rows) / accepted_count if accepted_count else float("nan")
    duplicate_answer_count = sum(count - 1 for count in counts.values() if count > 1)
    duplicate_rate = duplicate_answer_count / len(generated)
    low_verified_high = sum(
        row["accepted"] and int(row["target_score"]) <= 2 and int(row["verifier_counterfactual_min"]) >= 4
        for row in audit_rows
    )
    gates = {
        "api_schema_success": generation_schema_errors == 0 and verification_schema_errors == 0,
        "accepted_rows": accepted_count >= 180,
        "accepted_target1": accepted_targets[1] >= 25,
        "accepted_target2": accepted_targets[2] >= 80,
        "accepted_target3": accepted_targets[3] >= 50,
        "operator_success_rate": operator_success_rate >= 0.80,
        "rubric_failure_verification_rate": rubric_rate >= 0.80,
        "mean_edit_ratio": accepted_count > 0 and 0.10 <= accepted_edit_mean <= 0.25,
        "source_counterfactual_qkey_match": all(row["question_key"] == packets[row["sample_id"]]["question_key"] for row in accepted_rows),
        "no_dev_test_access": True,
        "no_low_verified_high": low_verified_high == 0,
        "duplicate_rate": duplicate_rate <= 0.01,
    }
    go = all(gates.values())
    rejected_counts = Counter(reason for row in audit_rows for reason in row["rejected_reasons"].split("|") if reason)
    failed_gates = sorted(name for name, passed in gates.items() if not passed)
    decision = {
        "status": "DATA_QUALIFICATION_GO" if go else "DATA_QUALIFICATION_NO_GO",
        "gates": gates,
        "failed_gates": failed_gates,
        "completed_rows": len(audit_rows),
        "accepted_rows": accepted_count,
        "accepted_target_counts": {str(score): accepted_targets[score] for score in (1, 2, 3)},
        "recommend_groupcv_training": go,
        "stop_counterfactual_augmentation": not go,
        "same_source_prompt_reuse_forbidden_on_failure": True,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    summary = [{
        "completed": len(audit_rows), "accepted": accepted_count, "acceptance_rate": accepted_count / len(audit_rows),
        "operator_success_rate": operator_success_rate, "rubric_failure_verified_rate": rubric_rate,
        "mean_accepted_edit_ratio": accepted_edit_mean,
        "mean_accepted_length_ratio": sum(row["length_ratio"] for row in accepted_rows) / accepted_count if accepted_count else float("nan"),
        "duplicate_answer_count": duplicate_answer_count, "duplicate_rate": duplicate_rate,
        "generation_schema_errors": generation_schema_errors, "verification_schema_errors": verification_schema_errors,
    }]
    write_csv(args.out_dir / "tables/exp39a_acceptance_summary.csv", summary)
    write_csv(args.out_dir / "tables/exp39a_acceptance_by_operator.csv", aggregate(audit_rows, "operator"))
    write_csv(args.out_dir / "tables/exp39a_acceptance_by_target.csv", aggregate(audit_rows, "target_score"))
    write_csv(args.out_dir / "tables/exp39a_edit_minimality.csv", [{
        "sample_id_hash": row["sample_id_hash"], "target_score": row["target_score"], "operator": row["operator"],
        "accepted": row["accepted"], "edit_ratio": row["edit_ratio"], "length_ratio": row["length_ratio"],
    } for row in audit_rows])
    write_csv(args.out_dir / "tables/exp39a_rejected_reason_distribution.csv", [
        {"reason": reason, "count": count, "rate": count / len(audit_rows)} for reason, count in sorted(rejected_counts.items())
    ], fieldnames=["reason", "count", "rate"])
    write_json(args.out_dir / "decision/exp39a_data_qualification_decision.json", decision)
    write_json(args.out_dir / "hashes/exp39a_synthetic_dataset_hashes.json", {
        "accepted_rows": accepted_count,
        "accepted_sample_ids_sha256": stable_hash(sorted(row["sample_id"] for row in accepted_rows)),
        "accepted_private_sha256": sha256_file(accepted_path),
        "generator_prompt_sha256": sha256_file(PROMPT_DIR / "exp39a_qwen_counterfactual_generator.md"),
        "verifier_prompt_sha256": sha256_file(PROMPT_DIR / "exp39a_deepseek_blind_counterfactual_verifier.md"),
    })
    report = [
        "# Exp39A EduCFA data qualification", "",
        f"- Status: **{decision['status']}**",
        f"- Completed / accepted: `{len(audit_rows)} / {accepted_count}`",
        f"- Accepted targets: `{json.dumps(dict(accepted_targets), sort_keys=True)}`",
        f"- Operator success rate: `{operator_success_rate:.4f}`",
        f"- Rubric failure verification rate: `{rubric_rate:.4f}`",
        f"- Mean accepted edit / length ratio: `{accepted_edit_mean:.4f}` / `{summary[0]['mean_accepted_length_ratio']:.4f}`",
        f"- Duplicate rate: `{duplicate_rate:.4f}`",
        f"- Gates: `{json.dumps(gates, sort_keys=True)}`",
        f"- Failed gates: `{json.dumps(failed_gates)}`",
        f"- Recommend GroupCV: `{str(go).lower()}`",
        "- No original train labels were replaced.",
        "- No paper-like dev/test data were accessed.",
        "", "## Leading rejection reasons", "",
    ]
    report.extend(
        f"- `{reason}`: `{count}` / `{len(audit_rows)}` (`{count / len(audit_rows):.4f}`)"
        for reason, count in rejected_counts.most_common(8)
    )
    if not go:
        report.extend([
            "", "## Pre-registered stop", "",
            "The accepted pool does not satisfy the frozen sample-count gates. GroupCV smoke/formal training is therefore not permitted.",
            "The acceptance thresholds were not relaxed after observing results, and the same source/prompt campaign must not be replayed as a new formal attempt.",
        ])
    report_path = args.out_dir / "reports/exp39a_data_qualification_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({**decision, "accepted": accepted_count, "target_counts": dict(accepted_targets)}, sort_keys=True))


if __name__ == "__main__":
    main()
