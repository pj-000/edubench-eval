"""Prepare Exp27D teacher-audit v4 consensus-calibration packets.

Exp27D keeps all 60 Exp27C re-pilot train samples for paired v3/v4 comparison
and adds 20 train-only stress cases. It prepares schema/prompt/packet artifacts
and leakage checks only. It does not call teacher APIs, train models, or use
dev/test labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import (  # noqa: E402
    clean,
    clamp_score,
    language,
    metric_name,
    read_jsonl,
    sample_id,
    subject,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_TRAIN = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_DEV = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_TEST = Path("thesis_exp/data/splits/question_seed42/test.jsonl")
DEFAULT_EXP27C_PACKETS = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27c_teacher_audit_v3_seed42/"
    "packets/exp27c_v3_repilot_blind_packets.jsonl"
)
DEFAULT_EXP27C_AUDIT_REFERENCE = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27c_teacher_audit_v3_seed42/"
    "packets/exp27c_v3_repilot_audit_reference_private.jsonl"
)
DEFAULT_EXP27C_SCORE_AGREEMENT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27c_teacher_audit_v3_seed42/"
    "tables/exp27c_v3_cross_provider_score_agreement.csv"
)
DEFAULT_EXP27C_RISK_AGREEMENT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27c_teacher_audit_v3_seed42/"
    "tables/exp27c_v3_cross_provider_risk_agreement.csv"
)
DEFAULT_EXP27C_CONFLICTS = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27c_teacher_audit_v3_seed42/"
    "tables/exp27c_v3_teacher_conflict_cases_light.csv"
)
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27d_teacher_audit_v4_seed42")
PROMPT_DIR = Path("thesis_exp/exp17_low_score_evidence/prompts")
SCHEMA_DIR = Path("thesis_exp/exp17_low_score_evidence/schemas")

SCHEMA_VERSION = "exp27d_teacher_audit_v4"
PROMPT_VERSION = "exp27d_v4"


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def file_sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def question_key(row: dict[str, Any]) -> str:
    return clean(row.get("question_key") or row.get("question_id"))


def teacher_metadata_text(row: dict[str, Any]) -> str:
    fields = {
        "language": language(row),
        "subject": subject(row),
        "education_level": clean(row.get("education_level_canonical") or row.get("education_level_raw")),
        "scenario": clean(row.get("scenario_canonical") or row.get("scenario_raw")),
        "metric_group": clean(row.get("metric_group")),
    }
    return json.dumps({key: value for key, value in fields.items() if value}, ensure_ascii=False, sort_keys=True)


def user_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": clean(row.get("question")),
        "answer": clean(row.get("answer")),
        "metric": metric_name(row),
        "rubric": clean(row.get("rubric")),
        "metadata": teacher_metadata_text(row),
    }


def row_signature(row: dict[str, Any]) -> tuple[str, str, str]:
    return (language(row), metric_name(row), subject(row))


def stratified_take(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if n <= 0 or not rows:
        return []
    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(row_signature(row), []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = list(buckets)
    rng.shuffle(keys)
    out: list[dict[str, Any]] = []
    while len(out) < n and keys:
        next_keys: list[tuple[str, str, str]] = []
        for key in keys:
            bucket = buckets[key]
            if bucket and len(out) < n:
                out.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return out[:n]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def label_from_row(row: dict[str, Any]) -> int:
    return clamp_score(row.get("label_5"))


def judge_scores(row: dict[str, Any]) -> list[int]:
    scores = row.get("judge_scores")
    if not isinstance(scores, dict):
        return []
    out: list[int] = []
    for value in scores.values():
        score = clamp_score(value)
        if 1 <= score <= 5:
            out.append(score)
    return out


def max_judge_score(row: dict[str, Any]) -> int:
    scores = judge_scores(row)
    return max(scores) if scores else label_from_row(row)


def min_judge_score(row: dict[str, Any]) -> int:
    scores = judge_scores(row)
    return min(scores) if scores else label_from_row(row)


def judge_score_spread(row: dict[str, Any]) -> int:
    scores = judge_scores(row)
    return (max(scores) - min(scores)) if scores else 0


def is_education_dimension_stress(row: dict[str, Any]) -> bool:
    text = " ".join(
        [
            metric_name(row),
            clean(row.get("metric_group")),
            clean(row.get("scenario_canonical") or row.get("scenario_raw")),
            clean(row.get("rubric")),
        ]
    ).lower()
    keywords = [
        "personal",
        "personalization",
        "scenario",
        "integration",
        "context",
        "higher-order",
        "higher order",
        "reasoning",
        "critical",
        "adapt",
    ]
    return any(keyword in text for keyword in keywords)


def packet_from_payload(
    sample_id_value: str,
    payload: dict[str, Any],
    source_meta: dict[str, Any],
    group: str,
    batch_size: int,
) -> dict[str, Any]:
    batch_index = int(packet_for.counter // batch_size) + 1
    packet_for.counter += 1
    return {
        "sample_id": sample_id_value,
        "split": "train",
        "pilot_group": group,
        "batch_id": f"exp27d_v4_repilot_{batch_index:03d}",
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "teacher_input": payload,
        "source_meta": {
            **source_meta,
            "sample_hash": sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        },
    }


def packet_for(row: dict[str, Any], group: str, batch_size: int) -> dict[str, Any]:
    sid = sample_id(row)
    payload = user_payload(row)
    return packet_from_payload(
        sid,
        payload,
        {
            "question_key": question_key(row),
            "language": language(row),
            "subject": subject(row),
            "metric": metric_name(row),
        },
        group,
        batch_size,
    )


packet_for.counter = 0  # type: ignore[attr-defined]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_id_guard(path: Path) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    qkeys: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ids.add(sample_id(row))
            qkeys.add(question_key(row))
    return ids, qkeys


def copy_protocol_files(out_dir: Path) -> dict[str, str]:
    files = {
        "blind_prompt": PROMPT_DIR / "exp27d_blind_teacher_prompt_v4.md",
        "audit_prompt": PROMPT_DIR / "exp27d_label_audit_prompt_v4.md",
        "blind_schema": SCHEMA_DIR / "exp27d_teacher_blind_schema_v4.json",
        "audit_schema": SCHEMA_DIR / "exp27d_teacher_audit_schema_v4.json",
    }
    targets = {
        "blind_prompt": out_dir / "prompts" / "exp27d_blind_teacher_prompt_v4.md",
        "audit_prompt": out_dir / "prompts" / "exp27d_label_audit_prompt_v4.md",
        "blind_schema": out_dir / "schema" / "exp27d_teacher_blind_schema_v4.json",
        "audit_schema": out_dir / "schema" / "exp27d_teacher_audit_schema_v4.json",
    }
    hashes: dict[str, str] = {}
    for key, source in files.items():
        target = targets[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[f"{key}_sha1"] = file_sha1(source)
    return hashes


def make_distribution_rows(packets: list[dict[str, Any]], ref_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        (
            packet["pilot_group"],
            ref_by_id[packet["sample_id"]]["original_score"],
            packet["source_meta"].get("language", ""),
            packet["source_meta"].get("metric", ""),
            packet["source_meta"].get("subject", ""),
        )
        for packet in packets
    )
    return [
        {
            "pilot_group": group,
            "label": label,
            "language": lang,
            "metric": metric,
            "subject": subj,
            "count": count,
        }
        for (group, label, lang, metric, subj), count in sorted(counts.items())
    ]


def take_ranked(rows: list[dict[str, Any]], n: int, seed: int, reverse: bool = True) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    rng = random.Random(seed)
    decorated = []
    for row in rows:
        decorated.append((judge_score_spread(row), max_judge_score(row), rng.random(), row))
    decorated.sort(reverse=reverse)
    return [row for *_rest, row in decorated[:n]]


def add_unique(
    selected: list[tuple[str, dict[str, Any]]],
    rows: list[dict[str, Any]],
    group: str,
    n: int,
    seen: set[str],
) -> None:
    for row in rows:
        if len([1 for g, _ in selected if g == group]) >= n:
            break
        sid = sample_id(row)
        if sid in seen:
            continue
        seen.add(sid)
        selected.append((group, row))


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.train_jsonl)
    row_by_id = {sample_id(row): row for row in rows}
    exp27c_packets = read_jsonl(args.exp27c_packets)
    exp27c_ref_rows = read_jsonl(args.exp27c_audit_reference)
    exp27c_ref_by_id = {str(row["sample_id"]): row for row in exp27c_ref_rows if row.get("sample_id")}
    if len(exp27c_packets) != 60:
        raise SystemExit(f"Expected 60 Exp27C packets, got {len(exp27c_packets)}")

    overlap_ids = {str(packet["sample_id"]) for packet in exp27c_packets}
    missing_refs = sorted(overlap_ids - set(exp27c_ref_by_id))
    if missing_refs:
        raise SystemExit(f"Exp27C audit reference missing sample ids: {missing_refs[:5]}")
    missing_train = sorted(overlap_ids - set(row_by_id))
    if missing_train:
        raise SystemExit(f"Exp27C packet ids missing from train split: {missing_train[:5]}")

    remaining = [row for row in rows if sample_id(row) not in overlap_ids]
    selected_stress: list[tuple[str, dict[str, Any]]] = []
    seen = set(overlap_ids)

    low_candidates = [
        row for row in remaining if label_from_row(row) <= 2 and (max_judge_score(row) >= 4 or judge_score_spread(row) >= 2)
    ]
    add_unique(
        selected_stress,
        take_ranked(low_candidates, args.low_stress_count, args.seed + 17),
        "new_low_hard_disagreement_stress",
        args.low_stress_count,
        seen,
    )

    high_candidates = [
        row for row in remaining if label_from_row(row) >= 4 and (min_judge_score(row) <= 2 or judge_score_spread(row) >= 2)
    ]
    add_unique(
        selected_stress,
        take_ranked(high_candidates, args.high_stress_count, args.seed + 31),
        "new_high_control_disagreement_stress",
        args.high_stress_count,
        seen,
    )

    mid_candidates = [
        row for row in remaining if label_from_row(row) == 3 and (max_judge_score(row) >= 4 or min_judge_score(row) <= 2)
    ]
    add_unique(
        selected_stress,
        take_ranked(mid_candidates, args.mid_stress_count, args.seed + 47),
        "new_label3_borderline_risk_stress",
        args.mid_stress_count,
        seen,
    )

    edu_candidates = [row for row in remaining if is_education_dimension_stress(row)]
    add_unique(
        selected_stress,
        stratified_take(edu_candidates, args.edu_stress_count * 3, args.seed + 61),
        "new_education_dimension_stress",
        args.edu_stress_count,
        seen,
    )

    target_stress = args.low_stress_count + args.high_stress_count + args.mid_stress_count + args.edu_stress_count
    if len(selected_stress) < target_stress:
        fallback = stratified_take(remaining, target_stress * 3, args.seed + 73)
        add_unique(selected_stress, fallback, "new_train_stress_fallback", target_stress - len(selected_stress), seen)
    if len(selected_stress) != target_stress:
        raise SystemExit(f"Expected {target_stress} new stress rows, got {len(selected_stress)}")

    packet_for.counter = 0  # type: ignore[attr-defined]
    packets: list[dict[str, Any]] = []
    audit_reference: list[dict[str, Any]] = []
    for old_packet in exp27c_packets:
        sid = str(old_packet["sample_id"])
        payload = dict(old_packet["teacher_input"])
        source_meta = dict(old_packet["source_meta"])
        source_meta["exp27c_pilot_group"] = old_packet.get("pilot_group", "")
        packet = packet_from_payload(sid, payload, source_meta, "exp27c_v3_60_paired_repilot", args.batch_size)
        packets.append(packet)
        ref = exp27c_ref_by_id[sid]
        audit_reference.append(
            {
                "sample_id": sid,
                "split": "train",
                "pilot_group": packet["pilot_group"],
                "batch_id": packet["batch_id"],
                "original_score": clamp_score(ref.get("original_score")),
                "source_meta": packet["source_meta"],
            }
        )
    for group, row in selected_stress:
        packet = packet_for(row, group, args.batch_size)
        packets.append(packet)
        audit_reference.append(
            {
                "sample_id": packet["sample_id"],
                "split": "train",
                "pilot_group": packet["pilot_group"],
                "batch_id": packet["batch_id"],
                "original_score": label_from_row(row),
                "source_meta": packet["source_meta"],
            }
        )

    out = args.out_dir
    write_jsonl(out / "packets" / "exp27d_v4_repilot_blind_packets.jsonl", packets)
    write_jsonl(out / "packets" / "exp27d_v4_repilot_audit_reference_private.jsonl", audit_reference)
    protocol_hashes = copy_protocol_files(out)

    ref_by_id = {row["sample_id"]: row for row in audit_reference}
    write_csv(out / "tables" / "exp27d_v4_candidate_distribution.csv", make_distribution_rows(packets, ref_by_id))

    dev_ids, dev_qkeys = load_id_guard(args.dev_jsonl)
    test_ids, test_qkeys = load_id_guard(args.test_jsonl)
    packet_ids = {packet["sample_id"] for packet in packets}
    packet_qkeys = {packet["source_meta"]["question_key"] for packet in packets}
    leakage_rows = [
        {"check": "dev_sample_overlap", "count": len(packet_ids & dev_ids)},
        {"check": "dev_question_overlap", "count": len(packet_qkeys & dev_qkeys)},
        {"check": "test_sample_overlap", "count": len(packet_ids & test_ids)},
        {"check": "test_question_overlap", "count": len(packet_qkeys & test_qkeys)},
        {"check": "test_label_read", "count": 0},
    ]
    write_csv(out / "tables" / "exp27d_v4_leakage_audit.csv", leakage_rows)

    decision = {
        "recommendation": "run_80_row_v4_dual_teacher_api_repilot_before_361",
        "packet_rows": len(packets),
        "exp27c_paired_overlap_rows": len(exp27c_packets),
        "new_stress_rows": len(selected_stress),
        "new_low_stress_rows": sum(1 for group, _ in selected_stress if group == "new_low_hard_disagreement_stress"),
        "new_high_stress_rows": sum(1 for group, _ in selected_stress if group == "new_high_control_disagreement_stress"),
        "new_mid_stress_rows": sum(1 for group, _ in selected_stress if group == "new_label3_borderline_risk_stress"),
        "new_education_dimension_stress_rows": sum(1 for group, _ in selected_stress if group == "new_education_dimension_stress"),
        "new_fallback_stress_rows": sum(1 for group, _ in selected_stress if group == "new_train_stress_fallback"),
        "batch_size": args.batch_size,
        "batch_count": len({packet["batch_id"] for packet in packets}),
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "test_label_read": False,
        "dev_test_used_for_id_guard_only": True,
        **protocol_hashes,
    }
    write_json(out / "decision" / "exp27d_teacher_audit_v4_prepare_decision.json", decision)

    report = [
        "# Exp27D Teacher Audit V4 Re-Pilot Preparation",
        "",
        "This step prepares the v4 consensus-calibration protocol before scaling. It does not call APIs or train.",
        "",
        "## Counts",
        "",
        f"- v4 re-pilot packets: {len(packets)}",
        f"- Exp27C paired overlap rows: {len(exp27c_packets)}",
        f"- new stress rows: {len(selected_stress)}",
        f"- new low hard-disagreement stress rows: {decision['new_low_stress_rows']}",
        f"- new high-control disagreement stress rows: {decision['new_high_stress_rows']}",
        f"- new label-3 borderline stress rows: {decision['new_mid_stress_rows']}",
        f"- new education-dimension stress rows: {decision['new_education_dimension_stress_rows']}",
        f"- fallback stress rows: {decision['new_fallback_stress_rows']}",
        "",
        "## V4 Consensus-Calibration Changes",
        "",
        "- Blind schema adds `surface_plausibility`.",
        "- Collector derives `failure_bucket` and `derived_overestimation_risk` instead of trusting raw risk alone.",
        "- Audit schema no longer asks the teacher to copy the blind object; it references blind id/hash only.",
        "- Audit schema separates soft/hard strictness and leniency disagreements.",
        "- Exact-or-missing evidence discipline is retained.",
        "",
        "## Guardrails",
        "",
        "- Blind packets do not contain original scores.",
        "- Blind packets do not contain recovered human reasons.",
        "- All packets are train-only; dev/test are excluded from packet construction.",
        "- Dev/test are read only for sample_id/question_key leakage guards.",
        "- Test labels are not read.",
        "- No API call or model training is performed in preparation.",
    ]
    write_text(out / "reports" / "exp27d_teacher_audit_v4_prepare_report.md", "\n".join(report))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Exp27D teacher-audit v4 re-pilot packets.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--exp27c-packets", type=Path, default=DEFAULT_EXP27C_PACKETS)
    parser.add_argument("--exp27c-audit-reference", type=Path, default=DEFAULT_EXP27C_AUDIT_REFERENCE)
    parser.add_argument("--exp27c-score-agreement", type=Path, default=DEFAULT_EXP27C_SCORE_AGREEMENT)
    parser.add_argument("--exp27c-risk-agreement", type=Path, default=DEFAULT_EXP27C_RISK_AGREEMENT)
    parser.add_argument("--exp27c-conflicts", type=Path, default=DEFAULT_EXP27C_CONFLICTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-stress-count", type=int, default=8)
    parser.add_argument("--high-stress-count", type=int, default=4)
    parser.add_argument("--mid-stress-count", type=int, default=4)
    parser.add_argument("--edu-stress-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
