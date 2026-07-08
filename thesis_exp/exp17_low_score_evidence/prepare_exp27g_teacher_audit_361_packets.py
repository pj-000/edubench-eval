"""Prepare Exp27G 361-case teacher-audit expansion packets.

Exp27G scales the Exp27D v4 teacher-audit protocol from an 80-case pilot to a
train-only 361-case controlled expansion. It prepares packets only: no API
calls, no training, and no dev/test label reads.
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
DEFAULT_EXP27F_ADJUDICATIONS = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27f_conflict_adjudication_pilot_seed42/"
    "annotation/exp27f_top40_adjudications.jsonl"
)
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27g_teacher_audited_361_seed42")
PROMPT_DIR = Path("thesis_exp/exp17_low_score_evidence/prompts")
SCHEMA_DIR = Path("thesis_exp/exp17_low_score_evidence/schemas")

SCHEMA_VERSION = "exp27d_teacher_audit_v4"
PROMPT_VERSION = "exp27d_v4"


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def file_sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def question_key(row: dict[str, Any]) -> str:
    return clean(row.get("question_key") or row.get("question_id"))


def label_from_row(row: dict[str, Any]) -> int:
    return clamp_score(row.get("label_5"))


def judge_scores(row: dict[str, Any]) -> list[int]:
    scores = row.get("judge_scores")
    if not isinstance(scores, dict):
        return []
    out: list[int] = []
    for value in scores.values():
        try:
            out.append(clamp_score(value))
        except Exception:  # noqa: BLE001
            continue
    return [score for score in out if 1 <= score <= 5]


def max_judge_score(row: dict[str, Any]) -> int:
    scores = judge_scores(row)
    return max(scores) if scores else label_from_row(row)


def min_judge_score(row: dict[str, Any]) -> int:
    scores = judge_scores(row)
    return min(scores) if scores else label_from_row(row)


def judge_score_spread(row: dict[str, Any]) -> int:
    scores = judge_scores(row)
    return (max(scores) - min(scores)) if scores else 0


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


def row_signature(row: dict[str, Any]) -> tuple[str, str, str]:
    return (language(row), metric_name(row), subject(row))


def ranked(rows: list[dict[str, Any]], seed: int, *, reverse: bool = True) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    decorated = []
    for row in rows:
        label = label_from_row(row)
        spread = judge_score_spread(row)
        risk_rank = spread
        if label <= 2:
            risk_rank += max_judge_score(row)
        elif label >= 4:
            risk_rank += 6 - min_judge_score(row)
        else:
            risk_rank += max_judge_score(row) - min_judge_score(row)
        decorated.append((risk_rank, spread, rng.random(), row))
    decorated.sort(reverse=reverse)
    return [row for *_rest, row in decorated]


def stratified_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(row_signature(row), []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = list(buckets)
    rng.shuffle(keys)
    out: list[dict[str, Any]] = []
    while keys:
        next_keys: list[tuple[str, str, str]] = []
        for key in keys:
            bucket = buckets[key]
            if bucket:
                out.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return out


def load_id_guard(path: Path) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    qkeys: set[str] = set()
    for row in read_jsonl(path):
        sid = sample_id(row)
        qkey = question_key(row)
        if sid:
            ids.add(sid)
        if qkey:
            qkeys.add(qkey)
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


def add_unique(
    selected: list[tuple[str, dict[str, Any]]],
    rows: list[dict[str, Any]],
    group: str,
    limit: int,
    seen: set[str],
) -> None:
    if limit <= 0:
        return
    current = sum(1 for existing_group, _ in selected if existing_group == group)
    for row in rows:
        if current >= limit:
            break
        sid = sample_id(row)
        if sid in seen:
            continue
        seen.add(sid)
        selected.append((group, row))
        current += 1


def add_until_total(
    selected: list[tuple[str, dict[str, Any]]],
    rows: list[dict[str, Any]],
    group: str,
    target_total: int,
    seen: set[str],
) -> None:
    for row in rows:
        if len(selected) >= target_total:
            break
        sid = sample_id(row)
        if sid in seen:
            continue
        seen.add(sid)
        selected.append((group, row))


def packet_for(row: dict[str, Any], group: str, idx: int, batch_size: int) -> dict[str, Any]:
    payload = user_payload(row)
    batch_id = f"exp27g_361_{idx // batch_size + 1:03d}"
    return {
        "sample_id": sample_id(row),
        "split": "train",
        "pilot_group": group,
        "batch_id": batch_id,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "teacher_input": payload,
        "source_meta": {
            "question_key": question_key(row),
            "language": language(row),
            "subject": subject(row),
            "metric": metric_name(row),
            "sample_hash": sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        },
    }


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


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.train_jsonl)
    row_by_id = {sample_id(row): row for row in rows}
    exp27f_rows = read_jsonl(args.exp27f_adjudications)
    exp27f_ids = [str(row["sample_id"]) for row in exp27f_rows]
    missing = sorted(set(exp27f_ids) - set(row_by_id))
    if missing:
        raise SystemExit(f"Exp27F ids missing from train split: {missing[:5]}")

    selected: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for sid in exp27f_ids:
        seen.add(sid)
        selected.append(("exp27f_top40_conflict_reaudit", row_by_id[sid]))

    low_all = [row for row in rows if label_from_row(row) <= 2]
    add_unique(selected, ranked(low_all, args.seed + 11), "train_all_low_label", args.low_count, seen)

    high_conflict = [
        row for row in rows if label_from_row(row) >= 4 and (min_judge_score(row) <= 2 or judge_score_spread(row) >= 2)
    ]
    add_unique(
        selected,
        ranked(high_conflict, args.seed + 23),
        "train_high_disagreement_protection",
        args.high_conflict_count,
        seen,
    )

    mid_borderline = [
        row
        for row in rows
        if label_from_row(row) == 3
        and (min_judge_score(row) <= 2 or max_judge_score(row) >= 4 or judge_score_spread(row) >= 2)
    ]
    add_unique(
        selected,
        ranked(mid_borderline, args.seed + 37),
        "train_mid_borderline",
        args.mid_count,
        seen,
    )

    edu_stress = [row for row in rows if is_education_dimension_stress(row)]
    add_unique(
        selected,
        stratified_rows(edu_stress, args.seed + 41),
        "train_education_dimension_stress",
        args.edu_count,
        seen,
    )

    high_clean = [
        row
        for row in rows
        if label_from_row(row) >= 4
        and judge_score_spread(row) <= 1
        and min_judge_score(row) >= 4
    ]
    add_until_total(
        selected,
        stratified_rows(high_clean, args.seed + 53),
        "train_clean_high_controls",
        args.total_count,
        seen,
    )

    if len(selected) < args.total_count:
        add_until_total(
            selected,
            stratified_rows(rows, args.seed + 67),
            "train_stratified_fallback",
            args.total_count,
            seen,
        )
    if len(selected) != args.total_count:
        raise SystemExit(f"Expected {args.total_count} selected rows, got {len(selected)}")

    packets: list[dict[str, Any]] = []
    audit_reference: list[dict[str, Any]] = []
    for idx, (group, row) in enumerate(selected):
        packet = packet_for(row, group, idx, args.batch_size)
        packets.append(packet)
        audit_reference.append(
            {
                "sample_id": packet["sample_id"],
                "split": "train",
                "pilot_group": group,
                "batch_id": packet["batch_id"],
                "original_score": label_from_row(row),
                "human_1": row.get("human_1_5"),
                "human_2": row.get("human_2_5"),
                "human_3": row.get("human_3_5"),
                "human_mean_5": row.get("human_mean_5"),
                "source_meta": packet["source_meta"],
            }
        )

    out = args.out_dir
    write_jsonl(out / "packets" / "exp27d_v4_repilot_blind_packets.jsonl", packets)
    write_jsonl(out / "packets" / "exp27d_v4_repilot_audit_reference_private.jsonl", audit_reference)
    write_jsonl(out / "packets" / "exp27g_361_teacher_packets.jsonl", packets)
    write_jsonl(out / "packets" / "exp27g_361_audit_reference_private.jsonl", audit_reference)
    protocol_hashes = copy_protocol_files(out)

    ref_by_id = {row["sample_id"]: row for row in audit_reference}
    write_csv(out / "tables" / "exp27g_sampling_distribution.csv", make_distribution_rows(packets, ref_by_id))

    dev_ids, dev_qkeys = load_id_guard(args.dev_jsonl)
    test_ids, test_qkeys = load_id_guard(args.test_jsonl)
    packet_ids = {packet["sample_id"] for packet in packets}
    packet_qkeys = {packet["source_meta"]["question_key"] for packet in packets}
    leakage_rows = [
        {"check": "packet_count", "count": len(packet_ids)},
        {"check": "dev_sample_overlap", "count": len(packet_ids & dev_ids)},
        {"check": "dev_question_overlap", "count": len(packet_qkeys & dev_qkeys)},
        {"check": "test_sample_overlap", "count": len(packet_ids & test_ids)},
        {"check": "test_question_overlap", "count": len(packet_qkeys & test_qkeys)},
        {"check": "test_label_read", "count": 0},
    ]
    write_csv(out / "tables" / "exp27g_leakage_audit.csv", leakage_rows)

    group_counts = Counter(group for group, _row in selected)
    label_counts = Counter(label_from_row(row) for _group, row in selected)
    decision = {
        "recommendation": "run_361_dual_teacher_api_then_collect_conflicts_for_adjudication",
        "packet_rows": len(packets),
        "batch_size": args.batch_size,
        "batch_count": len({packet["batch_id"] for packet in packets}),
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "group_counts": dict(sorted(group_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "test_label_read": False,
        "dev_test_used_for_id_guard_only": True,
        "proceed_to_api": all(int(row["count"]) == 0 for row in leakage_rows if row["check"] != "packet_count"),
        **protocol_hashes,
    }
    write_json(out / "decision" / "exp27g_prepare_decision.json", decision)

    report = [
        "# Exp27G Teacher-Audited 361 Preparation",
        "",
        "This step prepares a train-only 361-case teacher-audit expansion. It does not call APIs or train.",
        "",
        "## Counts",
        "",
        f"- packets: {len(packets)}",
        f"- batch_count: {decision['batch_count']}",
        f"- label_counts: `{dict(sorted(label_counts.items()))}`",
        f"- group_counts: `{dict(sorted(group_counts.items()))}`",
        "",
        "## Sampling Strategy",
        "",
        "- Start with Exp27F top40 conflict cases for re-audit.",
        "- Include all available train low-label cases after de-duplication.",
        "- Add high-label disagreement cases for high-score protection.",
        "- Add label-3 borderline cases.",
        "- Add education/rubric-dimension stress cases.",
        "- Fill remaining rows with clean high controls.",
        "",
        "## Guardrails",
        "",
        "- Blind packets contain no original score and no recovered human reason.",
        "- Audit reference contains train-only original scores for audit stage.",
        "- Dev/test are used only for sample_id/question_key leakage guards.",
        "- Test labels are not read.",
        "- No API call or model training is performed in preparation.",
    ]
    write_text(out / "reports" / "exp27g_prepare_report.md", "\n".join(report))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Exp27G 361-case teacher-audit packets.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--exp27f-adjudications", type=Path, default=DEFAULT_EXP27F_ADJUDICATIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-count", type=int, default=361)
    parser.add_argument("--low-count", type=int, default=111)
    parser.add_argument("--high-conflict-count", type=int, default=80)
    parser.add_argument("--mid-count", type=int, default=70)
    parser.add_argument("--edu-count", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
