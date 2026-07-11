"""Collect Exp28 teacher-protocol metrics without reading paper dev/test."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INVENTORY_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28b_paper_train_annotation_inventory_seed42"
)
DEFAULT_REFERENCE = DEFAULT_INVENTORY_DIR / "private" / "exp28b_benchmark_reference_2654.jsonl"
DEFAULT_PREDICTION_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42/private"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42"
)
PROVIDERS = ("qwen", "deepseek")
PROTOCOLS = (
    "p0_holistic_zero_shot",
    "p1_rubric_first",
    "p2_rubric_verify_then_score",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def qwk(gold: list[int], pred: list[int]) -> float:
    if not gold:
        return float("nan")
    n = len(gold)
    observed = [[0] * 5 for _ in range(5)]
    gold_counts = [0] * 5
    pred_counts = [0] * 5
    for left, right in zip(gold, pred):
        observed[left - 1][right - 1] += 1
        gold_counts[left - 1] += 1
        pred_counts[right - 1] += 1
    weighted_observed = 0.0
    weighted_expected = 0.0
    for left in range(5):
        for right in range(5):
            weight = ((left - right) ** 2) / 16.0
            weighted_observed += weight * observed[left][right]
            weighted_expected += weight * gold_counts[left] * pred_counts[right] / n
    return 1.0 - weighted_observed / weighted_expected if weighted_expected else 0.0


def kendall_tau_b(gold: list[int], pred: list[int]) -> float:
    concordant = discordant = tie_gold = tie_pred = 0
    for left in range(len(gold)):
        for right in range(left + 1, len(gold)):
            delta_gold = gold[left] - gold[right]
            delta_pred = pred[left] - pred[right]
            if delta_gold == 0 and delta_pred == 0:
                continue
            if delta_gold == 0:
                tie_gold += 1
            elif delta_pred == 0:
                tie_pred += 1
            elif delta_gold * delta_pred > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tie_gold) * (concordant + discordant + tie_pred)
    )
    return (concordant - discordant) / denominator if denominator else 0.0


def bin_value(score: int) -> str:
    return "low" if score <= 2 else "mid" if score == 3 else "high"


def metric_row(provider: str, protocol: str, subset: str, aligned: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    gold = [int(reference["original_label"]) for reference, _ in aligned]
    pred = [int(output["annotation"]["score"]) for _, output in aligned]
    n = len(gold)
    low_total = sum(value <= 2 for value in gold)
    high_total = sum(value >= 4 for value in gold)
    unanimous_pairs = [
        (int(reference["original_label"]), int(output["annotation"]["score"]))
        for reference, output in aligned
        if reference.get("human_agreement") == "unanimous"
    ]
    human_range_pairs = [
        (list(reference.get("human_scores") or []), int(output["annotation"]["score"]))
        for reference, output in aligned
        if reference.get("human_scores")
    ]
    row: dict[str, Any] = {
        "provider": provider,
        "protocol": protocol,
        "subset": subset,
        "n": n,
        "MAE": sum(abs(left - right) for left, right in zip(gold, pred)) / n if n else "",
        "Signed_Bias": sum(right - left for left, right in zip(gold, pred)) / n if n else "",
        "Exact_Match": sum(left == right for left, right in zip(gold, pred)) / n if n else "",
        "QWK": qwk(gold, pred) if n else "",
        "Kendall_tau_b": kendall_tau_b(gold, pred) if n else "",
        "Bin_Agreement": sum(bin_value(left) == bin_value(right) for left, right in zip(gold, pred)) / n if n else "",
        "unanimous_n": len(unanimous_pairs),
        "MAE_unanimous": (
            sum(abs(left - right) for left, right in unanimous_pairs) / len(unanimous_pairs)
            if unanimous_pairs
            else ""
        ),
        "Exact_Match_unanimous": (
            sum(left == right for left, right in unanimous_pairs) / len(unanimous_pairs)
            if unanimous_pairs
            else ""
        ),
        "within_human_score_range_rate": (
            sum(min(scores) <= prediction <= max(scores) for scores, prediction in human_range_pairs)
            / len(human_range_pairs)
            if human_range_pairs
            else ""
        ),
        "low_to_high_count": sum(left <= 2 and right >= 4 for left, right in zip(gold, pred)),
        "low_to_high_rate": (
            sum(left <= 2 and right >= 4 for left, right in zip(gold, pred)) / low_total if low_total else ""
        ),
        "high_to_low_count": sum(left >= 4 and right <= 2 for left, right in zip(gold, pred)),
        "high_to_low_rate": (
            sum(left >= 4 and right <= 2 for left, right in zip(gold, pred)) / high_total if high_total else ""
        ),
    }
    for score in range(1, 6):
        count = sum(value == score for value in gold)
        row[f"label{score}_recall"] = (
            sum(left == score and right == score for left, right in zip(gold, pred)) / count if count else ""
        )
    return row


def structure_row(provider: str, protocol: str, subset: str, outputs: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    valid = [row for row in outputs if isinstance(row.get("annotation"), dict) and not row.get("schema_errors")]
    annotations = [row["annotation"] for row in valid]
    caps = [row.get("score_cap") for row in annotations]
    failures = [row.get("major_failures") or [] for row in annotations]
    assessments = [row.get("rubric_assessment") or [] for row in annotations]
    return {
        "provider": provider,
        "protocol": protocol,
        "subset": subset,
        "expected_rows": expected,
        "output_rows": len(outputs),
        "valid_rows": len(valid),
        "parse_schema_success_rate": len(valid) / expected if expected else 0.0,
        "reason_nonempty_rate": sum(len(str(row.get("reason") or "").strip()) >= 20 for row in annotations) / len(annotations) if annotations else 0.0,
        "rubric_assessment_nonempty_rate": sum(bool(value) for value in assessments) / len(assessments) if assessments else 0.0,
        "major_failure_nonempty_rate": sum(value != ["no_major_failure"] for value in failures) / len(failures) if failures else 0.0,
        "score_cap_nonnull_rate": sum(value is not None for value in caps) / len(caps) if caps else 0.0,
        "score_cap_inconsistency_rate": (
            sum(
                annotation.get("score_cap") is not None
                and int(annotation["score"]) > int(annotation["score_cap"])
                for annotation in annotations
            )
            / len(annotations)
            if annotations
            else 0.0
        ),
        "mean_confidence": sum(float(row.get("confidence", 0.0)) for row in annotations) / len(annotations) if annotations else 0.0,
    }


def cross_provider_rows(outputs: dict[tuple[str, str], list[dict[str, Any]]], subset: str) -> list[dict[str, Any]]:
    rows = []
    for protocol in PROTOCOLS:
        by_provider = {
            provider: {
                str(row["sample_id"]): int(row["annotation"]["score"])
                for row in outputs.get((provider, protocol), [])
                if isinstance(row.get("annotation"), dict) and not row.get("schema_errors")
            }
            for provider in PROVIDERS
        }
        ids = sorted(set(by_provider["qwen"]) & set(by_provider["deepseek"]))
        left = [by_provider["qwen"][sid] for sid in ids]
        right = [by_provider["deepseek"][sid] for sid in ids]
        rows.append(
            {
                "protocol": protocol,
                "subset": subset,
                "n": len(ids),
                "score_exact_agreement": sum(a == b for a, b in zip(left, right)) / len(ids) if ids else "",
                "score_within1_agreement": sum(abs(a - b) <= 1 for a, b in zip(left, right)) / len(ids) if ids else "",
                "score_MAE": sum(abs(a - b) for a, b in zip(left, right)) / len(ids) if ids else "",
                "score_QWK": qwk(left, right) if ids else "",
            }
        )
    return rows


def choose_protocol(metrics: list[dict[str, Any]], structures: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in metrics if row["n"] > 0]
    if len(complete) < len(PROVIDERS) * len(PROTOCOLS):
        return {
            "status": "WAITING_FOR_PROTOCOL_API",
            "selected_protocol": None,
            "reason": "All provider/protocol development outputs are required before selection.",
        }
    structure_by_key = {(row["provider"], row["protocol"]): row for row in structures}
    baseline = {row["provider"]: row for row in complete if row["protocol"] == "p0_holistic_zero_shot"}
    protocol_scores = []
    for protocol in PROTOCOLS:
        rows = [row for row in complete if row["protocol"] == protocol]
        parse_ok = all(structure_by_key[(row["provider"], protocol)]["parse_schema_success_rate"] >= 0.98 for row in rows)
        guard_ok = all(
            float(row["MAE_unanimous"]) <= float(baseline[row["provider"]]["MAE_unanimous"]) + 0.05
            and float(row["high_to_low_rate"] or 0.0) <= float(baseline[row["provider"]]["high_to_low_rate"] or 0.0) + 0.02
            and float(row["Exact_Match_unanimous"]) >= float(baseline[row["provider"]]["Exact_Match_unanimous"]) - 0.03
            for row in rows
        )
        protocol_scores.append(
            {
                "protocol": protocol,
                "parse_ok": parse_ok,
                "guard_ok": guard_ok,
                "mean_MAE": sum(float(row["MAE"]) for row in rows) / len(rows),
                "mean_MAE_unanimous": sum(float(row["MAE_unanimous"]) for row in rows) / len(rows),
                "mean_low_to_high_rate": sum(float(row["low_to_high_rate"] or 0.0) for row in rows) / len(rows),
                "mean_QWK": sum(float(row["QWK"]) for row in rows) / len(rows),
                "mean_within_human_score_range_rate": sum(
                    float(row["within_human_score_range_rate"]) for row in rows
                ) / len(rows),
            }
        )
    eligible = [row for row in protocol_scores if row["parse_ok"] and row["guard_ok"]]
    if not eligible:
        return {
            "status": "NO_PROTOCOL_PASSES_GUARDS",
            "selected_protocol": None,
            "protocol_scores": protocol_scores,
            "reason": "Revise the annotation protocol before sealed qualification.",
        }
    selected = min(
        eligible,
        key=lambda row: (
            row["mean_low_to_high_rate"],
            row["mean_MAE_unanimous"],
            -row["mean_within_human_score_range_rate"],
            -row["mean_QWK"],
            row["protocol"],
        ),
    )
    return {
        "status": "READY_FOR_SEALED_QUALIFICATION",
        "selected_protocol": selected["protocol"],
        "protocol_scores": protocol_scores,
        "selection_rule": "min mean low-to-high subject to parse, unanimous-human MAE/exact, and high-to-low guards",
        "test_used": False,
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    references = {
        str(row["sample_id"]): row
        for row in read_jsonl(args.reference)
        if row.get("protocol_subset") == args.subset
    }
    expected = len(references)
    outputs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    metric_rows = []
    structure_rows = []
    for provider in PROVIDERS:
        for protocol in PROTOCOLS:
            path = args.prediction_dir / provider / protocol / f"{args.subset}.jsonl"
            rows = read_jsonl(path) if path.exists() else []
            outputs[(provider, protocol)] = rows
            valid_by_id = {
                str(row["sample_id"]): row
                for row in rows
                if isinstance(row.get("annotation"), dict) and not row.get("schema_errors")
            }
            aligned = [(reference, valid_by_id[sid]) for sid, reference in references.items() if sid in valid_by_id]
            metric_rows.append(metric_row(provider, protocol, args.subset, aligned))
            structure_rows.append(structure_row(provider, protocol, args.subset, rows, expected))

    cross_rows = cross_provider_rows(outputs, args.subset)
    fields = [
        "provider", "protocol", "subset", "n", "MAE", "Signed_Bias", "Exact_Match", "QWK",
        "Kendall_tau_b", "Bin_Agreement", "unanimous_n", "MAE_unanimous",
        "Exact_Match_unanimous", "within_human_score_range_rate", "low_to_high_count", "low_to_high_rate",
        "high_to_low_count", "high_to_low_rate", "label1_recall", "label2_recall", "label3_recall",
        "label4_recall", "label5_recall",
    ]
    write_csv(args.out_dir / "tables" / f"exp28c_{args.subset}_score_metrics.csv", metric_rows, fields)
    write_csv(
        args.out_dir / "tables" / f"exp28c_{args.subset}_structured_quality.csv",
        structure_rows,
        list(structure_rows[0]) if structure_rows else [],
    )
    write_csv(
        args.out_dir / "tables" / f"exp28c_{args.subset}_cross_provider_agreement.csv",
        cross_rows,
        list(cross_rows[0]) if cross_rows else [],
    )
    decision = choose_protocol(metric_rows, structure_rows)
    decision.update({"subset": args.subset, "reference_rows": expected, "dev_or_test_read": False})
    write_json(args.out_dir / "decision" / f"exp28c_{args.subset}_protocol_decision.json", decision)

    report_lines = [
        "# Exp28C Teacher Protocol Evaluation",
        "",
        f"- subset: `{args.subset}`",
        f"- reference rows: {expected}",
        f"- decision: **{decision['status']}**",
        f"- selected protocol: `{decision.get('selected_protocol')}`",
        "- paper dev/test read: no",
        "",
        "Protocol selection uses the paper-train protocol-development subset only. The benchmark",
        "human scores are evaluation references and never appear in blind teacher prompts.",
        "Independent teacher/model outputs are model annotations, not human review.",
    ]
    report_path = args.out_dir / "reports" / f"exp28c_{args.subset}_protocol_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--subset", choices=["protocol_development", "sealed_qualification"], default="protocol_development")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(collect(parse_args()), ensure_ascii=False, sort_keys=True))
