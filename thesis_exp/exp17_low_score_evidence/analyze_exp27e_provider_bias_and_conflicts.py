"""Analyze Exp27D provider bias and build Exp27E adjudication queues.

Exp27E is intentionally offline: it does not call teacher APIs, does not train,
and only reads Exp27D train-split packets plus ignored parsed teacher outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.collect_exp27d_teacher_audit_results import (  # noqa: E402
    derived_overestimation_risk,
    failure_bucket,
    training_use_compatible,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_jsonl, write_text  # noqa: E402


PROVIDERS = ("qwen", "deepseek")
DEFAULT_EXP27D_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27d_teacher_audit_v4_seed42")
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27e_provider_bias_conflict_analysis_seed42"
)
DEFAULT_TRAIN = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_DEV = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_TEST = Path("thesis_exp/data/splits/question_seed42/test.jsonl")
PREVIEW_CHARS = 240


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def label_region(label: int | None) -> str:
    if label is None:
        return "unknown"
    if label <= 2:
        return "low"
    if label == 3:
        return "mid"
    return "high"


def preview(value: Any, max_chars: int = PREVIEW_CHARS) -> str:
    text = " ".join(as_text(value).split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def load_split_ids(path: Path) -> tuple[set[str], set[str]]:
    """Extract only leakage guard identifiers from split files."""
    sample_ids: set[str] = set()
    question_keys: set[str] = set()
    for row in read_jsonl(path):
        sid = as_text(row.get("sample_id") or row.get("id") or row.get("record_id"))
        qkey = as_text(row.get("question_key") or row.get("question_id"))
        if sid:
            sample_ids.add(sid)
        if qkey:
            question_keys.add(qkey)
    return sample_ids, question_keys


def parsed_by_id(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = as_text(row.get("sample_id"))
        parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
        obj = parsed.get(key) if isinstance(parsed.get(key), dict) else {}
        if sid and obj:
            out[sid] = obj
    return out


def load_provider_outputs(exp27d_dir: Path) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    outputs: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for provider in PROVIDERS:
        blind_rows = read_jsonl(exp27d_dir / "annotations" / "parsed" / provider / "exp27d_blind_outputs.jsonl")
        audit_rows = read_jsonl(exp27d_dir / "annotations" / "parsed" / provider / "exp27d_audit_outputs.jsonl")
        outputs[provider] = {
            "blind": parsed_by_id(blind_rows, "blind"),
            "audit": parsed_by_id(audit_rows, "audit"),
        }
    return outputs


def load_packets(exp27d_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    packets = {
        as_text(row.get("sample_id")): row
        for row in read_jsonl(exp27d_dir / "packets" / "exp27d_v4_repilot_blind_packets.jsonl")
        if row.get("sample_id")
    }
    refs = {
        as_text(row.get("sample_id")): row
        for row in read_jsonl(exp27d_dir / "packets" / "exp27d_v4_repilot_audit_reference_private.jsonl")
        if row.get("sample_id")
    }
    return packets, refs


def provider_bias_rows(
    outputs: dict[str, dict[str, dict[str, dict[str, Any]]]],
    refs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        items: list[tuple[int, int]] = []
        for sid, ref in refs.items():
            label = as_int(ref.get("original_score"))
            score = as_int(outputs[provider]["blind"].get(sid, {}).get("teacher_score"))
            if label is not None and score is not None:
                items.append((label, score))
        diffs = [score - label for label, score in items]
        low_items = [(label, score) for label, score in items if label <= 2]
        high_items = [(label, score) for label, score in items if label >= 4]
        rows.append(
            {
                "provider": provider,
                "n": len(items),
                "mae_to_original": mean(abs(delta) for delta in diffs) if diffs else 0.0,
                "signed_bias_to_original": mean(diffs) if diffs else 0.0,
                "exact_agreement_to_original": sum(1 for delta in diffs if delta == 0) / len(diffs) if diffs else 0.0,
                "adjacent_agreement_to_original": sum(1 for delta in diffs if abs(delta) <= 1) / len(diffs)
                if diffs
                else 0.0,
                "low_label_mae": mean(abs(score - label) for label, score in low_items) if low_items else 0.0,
                "high_label_mae": mean(abs(score - label) for label, score in high_items) if high_items else 0.0,
                "low_to_high_teacher_count": sum(1 for label, score in items if label <= 2 and score >= 4),
                "high_to_low_teacher_count": sum(1 for label, score in items if label >= 4 and score <= 2),
            }
        )
    return rows


def provider_training_use_rows(
    outputs: dict[str, dict[str, dict[str, dict[str, Any]]]],
    packets: dict[str, dict[str, Any]],
    refs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        grouped: dict[str, list[str]] = defaultdict(list)
        for sid, audit in outputs[provider]["audit"].items():
            grouped[as_text(audit.get("recommended_training_use")) or "unknown"].append(sid)
        for use, ids in sorted(grouped.items()):
            by_region: Counter[str] = Counter()
            by_language: Counter[str] = Counter()
            for sid in ids:
                label = as_int(refs.get(sid, {}).get("original_score"))
                by_region[label_region(label)] += 1
                lang = as_text(packets.get(sid, {}).get("source_meta", {}).get("language")) or "unknown"
                by_language[lang] += 1
            rows.append(
                {
                    "provider": provider,
                    "recommended_training_use": use,
                    "count": len(ids),
                    "by_label_region": dict(sorted(by_region.items())),
                    "by_language": dict(sorted(by_language.items())),
                }
            )
    return rows


def provider_view(outputs: dict[str, dict[str, dict[str, dict[str, Any]]]], provider: str, sid: str) -> dict[str, Any]:
    blind = outputs[provider]["blind"].get(sid, {})
    audit = outputs[provider]["audit"].get(sid, {})
    return {
        "score": as_int(blind.get("teacher_score")),
        "failure_bucket": failure_bucket(blind),
        "derived_risk": derived_overestimation_risk(blind),
        "training_use": as_text(audit.get("recommended_training_use")) or "unknown",
        "hard_conflict": as_bool(audit.get("hard_conflict")),
        "needs_human_review": as_bool(audit.get("needs_human_review")),
        "label_quality": as_text(audit.get("label_quality")),
        "label_noise_type": as_text(audit.get("label_noise_type")),
        "teacher_score_audit": as_int(audit.get("teacher_score")),
    }


def conflict_types(label: int | None, q: dict[str, Any], d: dict[str, Any]) -> list[str]:
    types: list[str] = []
    q_score = q.get("score")
    d_score = d.get("score")
    if q_score is not None and d_score is not None:
        gap = abs(q_score - d_score)
        if gap >= 2:
            types.append("score_gap_ge2")
        if q_score >= d_score + 2:
            types.append("qwen_lenient_deepseek_strict")
        if d_score >= q_score + 2:
            types.append("deepseek_lenient_qwen_strict")
    if label is not None:
        if label <= 2 and ((q_score is not None and q_score >= 4) or (d_score is not None and d_score >= 4)):
            types.append("low_human_teacher_high")
        if label >= 4 and ((q_score is not None and q_score <= 2) or (d_score is not None and d_score <= 2)):
            types.append("high_human_teacher_low")
        q_disagree = q_score is not None and abs(q_score - label) >= 2
        d_disagree = d_score is not None and abs(d_score - label) >= 2
        if q_disagree and d_disagree:
            types.append("both_teachers_disagree_with_human")
        if q_disagree or d_disagree:
            types.append("possible_original_label_noise")
    if q.get("failure_bucket") != d.get("failure_bucket"):
        types.append("failure_bucket_disagreement")
    if q.get("derived_risk") != d.get("derived_risk"):
        types.append("derived_risk_disagreement")
    if label is not None and label >= 4 and (q.get("hard_conflict") or d.get("hard_conflict")):
        types.append("high_control_hard_conflict")
    if q.get("label_noise_type") != d.get("label_noise_type") or q.get("label_quality") != d.get("label_quality"):
        types.append("possible_teacher_strictness_difference")
    return sorted(set(types))


def priority_for(label: int | None, q: dict[str, Any], d: dict[str, Any], types: list[str]) -> int:
    q_score = q.get("score")
    d_score = d.get("score")
    gap = abs(q_score - d_score) if q_score is not None and d_score is not None else 0
    if label is not None and label <= 2 and "low_human_teacher_high" in types:
        return 1
    if label is not None and label >= 4 and "high_human_teacher_low" in types:
        return 2
    if gap >= 3:
        return 3
    if gap == 2:
        return 4
    if "failure_bucket_disagreement" in types:
        return 5
    if "derived_risk_disagreement" in types:
        return 6
    return 7


def build_conflict_queue(
    outputs: dict[str, dict[str, dict[str, dict[str, Any]]]],
    packets: dict[str, dict[str, Any]],
    refs: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    queue: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    distribution: Counter[str] = Counter()
    all_ids = sorted(set(refs) & set(packets))
    for sid in all_ids:
        label = as_int(refs[sid].get("original_score"))
        q = provider_view(outputs, "qwen", sid)
        d = provider_view(outputs, "deepseek", sid)
        types = conflict_types(label, q, d)
        if not types:
            continue
        for ctype in types:
            distribution[ctype] += 1
        priority = priority_for(label, q, d, types)
        row = packets[sid]
        teacher_input = row.get("teacher_input", {}) if isinstance(row.get("teacher_input"), dict) else {}
        source_meta = row.get("source_meta", {}) if isinstance(row.get("source_meta"), dict) else {}
        queue.append(
            {
                "sample_id": sid,
                "label": label,
                "label_region": label_region(label),
                "language": source_meta.get("language", ""),
                "pilot_group": refs[sid].get("pilot_group", row.get("pilot_group", "")),
                "qwen_score": q.get("score"),
                "deepseek_score": d.get("score"),
                "qwen_failure_bucket": q.get("failure_bucket"),
                "deepseek_failure_bucket": d.get("failure_bucket"),
                "qwen_derived_risk": q.get("derived_risk"),
                "deepseek_derived_risk": d.get("derived_risk"),
                "qwen_training_use": q.get("training_use"),
                "deepseek_training_use": d.get("training_use"),
                "conflict_type": ";".join(types),
                "adjudication_priority": priority,
                "top40_for_manual_review": False,
                "question_preview": preview(teacher_input.get("question")),
                "answer_preview": preview(teacher_input.get("answer")),
                "rubric_preview": preview(teacher_input.get("rubric")),
            }
        )
        packet_rows.append(
            {
                "sample_id": sid,
                "split": "train",
                "question_key": source_meta.get("question_key", ""),
                "original_train_score": label,
                "label_region": label_region(label),
                "language": source_meta.get("language", ""),
                "subject": source_meta.get("subject", ""),
                "metric": teacher_input.get("metric", ""),
                "question": teacher_input.get("question", ""),
                "answer": teacher_input.get("answer", ""),
                "rubric": teacher_input.get("rubric", ""),
                "metadata": teacher_input.get("metadata", ""),
                "compact_teacher_disagreement_summary": {
                    "qwen": {
                        "score": q.get("score"),
                        "failure_bucket": q.get("failure_bucket"),
                        "derived_risk": q.get("derived_risk"),
                        "training_use": q.get("training_use"),
                        "hard_conflict": q.get("hard_conflict"),
                    },
                    "deepseek": {
                        "score": d.get("score"),
                        "failure_bucket": d.get("failure_bucket"),
                        "derived_risk": d.get("derived_risk"),
                        "training_use": d.get("training_use"),
                        "hard_conflict": d.get("hard_conflict"),
                    },
                    "question_key": source_meta.get("question_key", ""),
                    "conflict_type": types,
                    "adjudication_priority": priority,
                },
                "top40_for_manual_review": False,
            }
        )
    queue.sort(
        key=lambda row: (
            int(row["adjudication_priority"]),
            -abs(int(row.get("qwen_score") or 0) - int(row.get("deepseek_score") or 0)),
            str(row["sample_id"]),
        )
    )
    top_ids = {row["sample_id"] for row in queue[:40]}
    for row in queue:
        row["top40_for_manual_review"] = row["sample_id"] in top_ids
    for row in packet_rows:
        row["top40_for_manual_review"] = row["sample_id"] in top_ids

    distribution_rows = [{"conflict_type": ctype, "count": count} for ctype, count in sorted(distribution.items())]
    return queue, packet_rows, distribution_rows


def map_training_use(use: str) -> str:
    if use == "high_weight":
        return "high_trust"
    if use == "low_weight":
        return "low_weight"
    if use == "exclude":
        return "excluded"
    return "review_only"


def selective_review_trigger(label: int | None, q: dict[str, Any], d: dict[str, Any]) -> bool:
    q_score = q.get("score")
    d_score = d.get("score")
    score_gap = q_score is not None and d_score is not None and abs(q_score - d_score) >= 2
    low_teacher_high = label is not None and label <= 2 and (
        (q_score is not None and q_score >= 4) or (d_score is not None and d_score >= 4)
    )
    high_teacher_low = label is not None and label >= 4 and (
        (q_score is not None and q_score <= 2) or (d_score is not None and d_score <= 2)
    )
    return any(
        [
            score_gap,
            low_teacher_high,
            high_teacher_low,
            q.get("failure_bucket") != d.get("failure_bucket"),
            q.get("derived_risk") != d.get("derived_risk"),
            q.get("hard_conflict"),
            d.get("hard_conflict"),
            not training_use_compatible(as_text(q.get("training_use")), as_text(d.get("training_use"))),
        ]
    )


def assign_policy(policy: str, label: int | None, q: dict[str, Any], d: dict[str, Any]) -> str:
    if policy == "qwen_only_primary":
        return map_training_use(as_text(q.get("training_use")))
    if policy == "deepseek_only_primary":
        return map_training_use(as_text(d.get("training_use")))
    if policy == "exact_or_adjacent_consensus_else_review":
        if q.get("training_use") == "exclude" or d.get("training_use") == "exclude":
            return "excluded"
        q_score = q.get("score")
        d_score = d.get("score")
        score_ok = q_score is not None and d_score is not None and abs(q_score - d_score) <= 1
        if (
            score_ok
            and q.get("failure_bucket") == d.get("failure_bucket")
            and q.get("derived_risk") == d.get("derived_risk")
            and training_use_compatible(as_text(q.get("training_use")), as_text(d.get("training_use")))
        ):
            if q.get("training_use") == "high_weight" and d.get("training_use") == "high_weight":
                return "high_trust"
            return "low_weight"
        return "review_only"
    if policy == "deepseek_primary_qwen_selective_review":
        base = map_training_use(as_text(d.get("training_use")))
        if base == "excluded":
            return "excluded"
        return "review_only" if selective_review_trigger(label, q, d) else base
    if policy == "qwen_primary_deepseek_selective_review":
        base = map_training_use(as_text(q.get("training_use")))
        if base == "excluded":
            return "excluded"
        return "review_only" if selective_review_trigger(label, q, d) else base
    if policy == "original_human_with_teacher_quality_weight":
        if q.get("training_use") == "exclude" or d.get("training_use") == "exclude":
            return "excluded"
        if selective_review_trigger(label, q, d):
            return "review_only"
        if q.get("training_use") == "high_weight" and d.get("training_use") == "high_weight":
            return "high_trust"
        return "low_weight"
    return "review_only"


def consensus_policy_rows(
    outputs: dict[str, dict[str, dict[str, dict[str, Any]]]],
    refs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    policies = [
        "qwen_only_primary",
        "deepseek_only_primary",
        "exact_or_adjacent_consensus_else_review",
        "deepseek_primary_qwen_selective_review",
        "qwen_primary_deepseek_selective_review",
        "original_human_with_teacher_quality_weight",
    ]
    rows: list[dict[str, Any]] = []
    for policy in policies:
        assignments: Counter[str] = Counter()
        low_total = 0
        high_total = 0
        low_review = 0
        high_review = 0
        remaining_high_conflicts = 0
        for sid, ref in refs.items():
            label = as_int(ref.get("original_score"))
            q = provider_view(outputs, "qwen", sid)
            d = provider_view(outputs, "deepseek", sid)
            assignment = assign_policy(policy, label, q, d)
            assignments[assignment] += 1
            if label is not None and label <= 2:
                low_total += 1
                low_review += int(assignment == "review_only")
            if label is not None and label >= 4:
                high_total += 1
                high_review += int(assignment == "review_only")
                if assignment in {"high_trust", "low_weight"} and (q.get("hard_conflict") or d.get("hard_conflict")):
                    remaining_high_conflicts += 1
        rows.append(
            {
                "policy": policy,
                "high_trust_count": assignments["high_trust"],
                "low_weight_count": assignments["low_weight"],
                "review_only_count": assignments["review_only"],
                "excluded_count": assignments["excluded"],
                "low_label_review_rate": low_review / low_total if low_total else 0.0,
                "high_label_review_rate": high_review / high_total if high_total else 0.0,
                "estimated_training_rows": assignments["high_trust"] + assignments["low_weight"],
                "high_control_hard_conflict_remaining": remaining_high_conflicts,
            }
        )
    return rows


def leakage_rows(
    packets: dict[str, dict[str, Any]],
    train_path: Path,
    dev_path: Path,
    test_path: Path,
) -> list[dict[str, Any]]:
    packet_ids = set(packets)
    packet_qkeys = {
        as_text(row.get("source_meta", {}).get("question_key"))
        for row in packets.values()
        if isinstance(row.get("source_meta"), dict) and row.get("source_meta", {}).get("question_key")
    }
    train_ids, train_qkeys = load_split_ids(train_path)
    dev_ids, dev_qkeys = load_split_ids(dev_path)
    test_ids, test_qkeys = load_split_ids(test_path)
    return [
        {"check": "packet_count", "count": len(packet_ids)},
        {"check": "packet_not_in_train_sample_count", "count": len(packet_ids - train_ids)},
        {"check": "packet_not_in_train_question_count", "count": len(packet_qkeys - train_qkeys)},
        {"check": "dev_sample_overlap", "count": len(packet_ids & dev_ids)},
        {"check": "dev_question_overlap", "count": len(packet_qkeys & dev_qkeys)},
        {"check": "test_sample_overlap", "count": len(packet_ids & test_ids)},
        {"check": "test_question_overlap", "count": len(packet_qkeys & test_qkeys)},
        {"check": "test_label_read", "count": 0},
    ]


def choose_primary(bias_rows: list[dict[str, Any]], outputs: dict[str, dict[str, dict[str, dict[str, Any]]]], refs: dict[str, dict[str, Any]]) -> str:
    by_provider = {row["provider"]: row for row in bias_rows}

    def high_conflict_count(provider: str) -> int:
        total = 0
        for sid, ref in refs.items():
            label = as_int(ref.get("original_score"))
            if label is not None and label >= 4 and provider_view(outputs, provider, sid).get("hard_conflict"):
                total += 1
        return total

    q = by_provider.get("qwen", {})
    d = by_provider.get("deepseek", {})
    if not q or not d:
        return "none_yet"
    deepseek_better = (
        d.get("low_to_high_teacher_count", 999) <= q.get("low_to_high_teacher_count", 999)
        and high_conflict_count("deepseek") <= high_conflict_count("qwen")
        and d.get("mae_to_original", 999.0) <= q.get("mae_to_original", 999.0) + 0.25
    )
    qwen_better = (
        q.get("low_to_high_teacher_count", 999) < d.get("low_to_high_teacher_count", 999)
        and high_conflict_count("qwen") < high_conflict_count("deepseek")
        and q.get("mae_to_original", 999.0) <= d.get("mae_to_original", 999.0) + 0.25
    )
    if deepseek_better and not qwen_better:
        return "deepseek"
    if qwen_better and not deepseek_better:
        return "qwen"
    return "none_yet"


def build_report(
    decision: dict[str, Any],
    bias_rows: list[dict[str, Any]],
    conflict_dist: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    queue_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Exp27E Provider Bias and Conflict-Adjudication Analysis",
        "",
        "Exp27E is an offline analysis. It does not call APIs, train models, or use GPU.",
        "",
        "## Decision",
        "",
        f"- recommend_use_both_for_361: {decision['recommend_use_both_for_361']}",
        f"- recommended_primary_teacher_for_full_train: `{decision['recommended_primary_teacher_for_full_train']}`",
        f"- recommend_selective_second_teacher: {decision['recommend_selective_second_teacher']}",
        f"- recommend_gpt55_or_human_adjudication: {decision['recommend_gpt55_or_human_adjudication']}",
        f"- proceed_to_361_after_adjudication: {decision['proceed_to_361_after_adjudication']}",
        "",
        "## Provider Bias vs Original Human Label",
        "",
        "| provider | n | MAE | signed bias | exact | adjacent | low->high | high->low |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bias_rows:
        lines.append(
            f"| {row['provider']} | {row['n']} | {float(row['mae_to_original']):.4f} | "
            f"{float(row['signed_bias_to_original']):.4f} | {float(row['exact_agreement_to_original']):.4f} | "
            f"{float(row['adjacent_agreement_to_original']):.4f} | {row['low_to_high_teacher_count']} | "
            f"{row['high_to_low_teacher_count']} |"
        )
    lines.extend(
        [
            "",
            "## Conflict Types",
            "",
            "| conflict_type | count |",
            "|---|---:|",
        ]
    )
    for row in conflict_dist:
        lines.append(f"| {row['conflict_type']} | {row['count']} |")
    lines.extend(
        [
            "",
            "## Consensus Policy Simulation",
            "",
            "| policy | high_trust | low_weight | review_only | excluded | estimated_train | high_conflicts_left |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in policy_rows:
        lines.append(
            f"| {row['policy']} | {row['high_trust_count']} | {row['low_weight_count']} | "
            f"{row['review_only_count']} | {row['excluded_count']} | {row['estimated_training_rows']} | "
            f"{row['high_control_hard_conflict_remaining']} |"
        )
    top_counts = Counter(int(row["adjudication_priority"]) for row in queue_rows if row.get("top40_for_manual_review"))
    lines.extend(
        [
            "",
            "## Adjudication Queue",
            "",
            f"- full_queue_size: {len(queue_rows)}",
            f"- top40_size: {sum(1 for row in queue_rows if row.get('top40_for_manual_review'))}",
            f"- top40_priority_counts: {dict(sorted(top_counts.items()))}",
            "",
            "The queue contains lightweight previews in CSV and train-only structured packets for later GPT5.5Pro or human adjudication.",
            "",
            "## Guardrails",
            "",
            "- no training",
            "- no API calls",
            "- no GPU",
            "- no dev labels are used",
            "- no test labels are read",
            "- dev/test samples are excluded from adjudication packets",
            "- raw API outputs and full parsed teacher text remain local/ignored",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    exp27d_dir: Path = args.exp27d_dir
    out_dir: Path = args.out_dir
    tables_dir = out_dir / "tables"
    reports_dir = out_dir / "reports"
    decision_dir = out_dir / "decision"
    annotation_dir = out_dir / "annotation"
    prompt_dir = out_dir / "prompts"
    schema_dir = out_dir / "schema"
    for path in [tables_dir, reports_dir, decision_dir, annotation_dir, prompt_dir, schema_dir]:
        path.mkdir(parents=True, exist_ok=True)

    packets, refs = load_packets(exp27d_dir)
    outputs = load_provider_outputs(exp27d_dir)
    missing = [
        f"{provider}/{stage}"
        for provider in PROVIDERS
        for stage in ("blind", "audit")
        if not outputs[provider][stage]
    ]
    if missing:
        raise FileNotFoundError(f"Missing parsed Exp27D annotations for: {', '.join(missing)}")

    bias_rows = provider_bias_rows(outputs, refs)
    training_rows = provider_training_use_rows(outputs, packets, refs)
    queue_rows, packet_rows, conflict_dist_rows = build_conflict_queue(outputs, packets, refs)
    policy_rows = consensus_policy_rows(outputs, refs)
    leak_rows = leakage_rows(packets, args.train_jsonl, args.dev_jsonl, args.test_jsonl)
    exp27d_decision = read_json(exp27d_dir / "decision" / "exp27d_teacher_audit_v4_api_repilot_decision.json")

    primary = choose_primary(bias_rows, outputs, refs)
    queue_size = len(queue_rows)
    high_control_conflict = float(exp27d_decision.get("high_control_hard_conflict_rate", 0.0) or 0.0)
    score_adjacent = float(exp27d_decision.get("teacher_score_adjacent_agreement", 0.0) or 0.0)
    failure_bucket_agreement = float(exp27d_decision.get("failure_bucket_agreement", 0.0) or 0.0)
    recommend_adjudication = queue_size > 0 and (
        queue_size >= 25 or high_control_conflict > 0.10 or score_adjacent < 0.95 or failure_bucket_agreement < 0.80
    )
    decision = {
        "recommend_use_both_for_361": True,
        "recommended_primary_teacher_for_full_train": primary,
        "recommend_selective_second_teacher": True,
        "recommend_gpt55_or_human_adjudication": recommend_adjudication,
        "proceed_to_361_after_adjudication": recommend_adjudication,
        "reason": (
            "Exp27D has stable parsing/schema but provider judgment disagreement remains high; "
            "use both teachers for 361 protocol validation and adjudicate top conflicts before scaling."
        ),
        "exp27d_teacher_score_adjacent_agreement": score_adjacent,
        "exp27d_failure_bucket_agreement": failure_bucket_agreement,
        "exp27d_high_control_hard_conflict_rate": high_control_conflict,
        "full_queue_size": queue_size,
        "top40_queue_size": sum(1 for row in queue_rows if row.get("top40_for_manual_review")),
    }

    write_csv(tables_dir / "exp27e_provider_vs_human_score_bias.csv", bias_rows)
    write_csv(tables_dir / "exp27e_provider_training_use_distribution.csv", training_rows)
    write_csv(tables_dir / "exp27e_conflict_type_distribution.csv", conflict_dist_rows)
    write_csv(tables_dir / "exp27e_consensus_policy_simulation.csv", policy_rows)
    write_csv(tables_dir / "exp27e_leakage_audit.csv", leak_rows)
    write_csv(annotation_dir / "exp27e_gpt55_human_adjudication_queue.csv", queue_rows)
    write_jsonl(annotation_dir / "exp27e_gpt55_human_adjudication_packets.jsonl", packet_rows)
    write_json(decision_dir / "exp27e_provider_bias_conflict_analysis_decision.json", decision)
    write_text(
        reports_dir / "exp27e_provider_bias_conflict_analysis_report.md",
        build_report(decision, bias_rows, conflict_dist_rows, policy_rows, queue_rows),
    )

    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Exp27E provider bias and conflict adjudication queues.")
    parser.add_argument("--exp27d-dir", type=Path, default=DEFAULT_EXP27D_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST)
    args = parser.parse_args()
    print(json.dumps(analyze(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
