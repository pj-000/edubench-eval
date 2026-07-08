"""Prepare Exp27B teacher-audit v2 re-pilot packets.

Exp27B revises the Exp27A teacher-audit protocol before scaling annotation.
It only prepares schema/prompt/packet artifacts and leakage checks. It does not
call teacher APIs, train models, or use dev/test labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
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
DEFAULT_EXP27A_AGREEMENT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27a_teacher_audit_pilot_seed42/"
    "tables/exp27a_teacher_blind_cross_provider_agreement.csv"
)
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27b_teacher_audit_v2_seed42")
PROMPT_DIR = Path("thesis_exp/exp17_low_score_evidence/prompts")
SCHEMA_DIR = Path("thesis_exp/exp17_low_score_evidence/schemas")

SCHEMA_VERSION = "exp27b_teacher_audit_v2"
PROMPT_VERSION = "exp27b_v2"


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


def exp27a_sample_ids(path: Path) -> list[str]:
    rows = read_csv_rows(path)
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        sid = clean(row.get("sample_id"))
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def packet_for(row: dict[str, Any], group: str, batch_size: int) -> dict[str, Any]:
    sid = sample_id(row)
    batch_index = int(packet_for.counter // batch_size) + 1
    packet_for.counter += 1
    payload = user_payload(row)
    return {
        "sample_id": sid,
        "split": "train",
        "pilot_group": group,
        "batch_id": f"exp27b_v2_repilot_{batch_index:03d}",
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
        "blind_prompt": PROMPT_DIR / "exp27b_blind_teacher_prompt_v2.md",
        "audit_prompt": PROMPT_DIR / "exp27b_label_audit_prompt_v2.md",
        "blind_schema": SCHEMA_DIR / "exp27b_teacher_blind_schema_v2.json",
        "audit_schema": SCHEMA_DIR / "exp27b_teacher_audit_schema_v2.json",
    }
    targets = {
        "blind_prompt": out_dir / "prompts" / "exp27b_blind_teacher_prompt_v2.md",
        "audit_prompt": out_dir / "prompts" / "exp27b_label_audit_prompt_v2.md",
        "blind_schema": out_dir / "schema" / "exp27b_teacher_blind_schema_v2.json",
        "audit_schema": out_dir / "schema" / "exp27b_teacher_audit_schema_v2.json",
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


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.train_jsonl)
    row_by_id = {sample_id(row): row for row in rows}
    old_ids = exp27a_sample_ids(args.exp27a_agreement)
    old_rows: list[dict[str, Any]] = []
    missing_old_ids: list[str] = []
    for sid in old_ids:
        row = row_by_id.get(sid)
        if row is None:
            missing_old_ids.append(sid)
        else:
            old_rows.append(row)
    if missing_old_ids:
        raise SystemExit(f"Exp27A sample ids missing from train split: {missing_old_ids[:5]}")

    old_set = {sample_id(row) for row in old_rows}
    remaining = [row for row in rows if sample_id(row) not in old_set]
    low_rows = [row for row in remaining if clamp_score(row.get("label_5")) <= 2]
    mid_rows = [row for row in remaining if clamp_score(row.get("label_5")) == 3]
    high_rows = [row for row in remaining if clamp_score(row.get("label_5")) >= 4]
    low_sel = stratified_take(low_rows, args.new_low_count, args.seed + 17)
    mid_sel = stratified_take(mid_rows, args.new_mid_count, args.seed + 31)
    high_sel = stratified_take(high_rows, args.new_high_count, args.seed + 47)

    selected: list[tuple[str, dict[str, Any]]] = [("exp27a_v1_repilot_overlap", row) for row in old_rows]
    selected.extend(("new_train_low_risk_focused", row) for row in low_sel)
    selected.extend(("new_train_label3_borderline", row) for row in mid_sel)
    selected.extend(("new_train_high_control", row) for row in high_sel)

    rng = random.Random(args.seed)
    old_part = selected[: len(old_rows)]
    new_part = selected[len(old_rows) :]
    rng.shuffle(new_part)
    selected = old_part + new_part

    packet_for.counter = 0  # type: ignore[attr-defined]
    packets = [packet_for(row, group, args.batch_size) for group, row in selected]
    audit_reference = [
        {
            "sample_id": packet["sample_id"],
            "split": "train",
            "pilot_group": packet["pilot_group"],
            "batch_id": packet["batch_id"],
            "original_score": clamp_score(row.get("label_5")),
            "source_meta": packet["source_meta"],
        }
        for packet, (_, row) in zip(packets, selected)
    ]

    out = args.out_dir
    write_jsonl(out / "packets" / "exp27b_v2_repilot_blind_packets.jsonl", packets)
    write_jsonl(out / "packets" / "exp27b_v2_repilot_audit_reference_private.jsonl", audit_reference)
    protocol_hashes = copy_protocol_files(out)

    ref_by_id = {row["sample_id"]: row for row in audit_reference}
    write_csv(out / "tables" / "exp27b_v2_repilot_candidate_distribution.csv", make_distribution_rows(packets, ref_by_id))

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
    write_csv(out / "tables" / "exp27b_v2_leakage_audit.csv", leakage_rows)

    decision = {
        "recommendation": "run_40_to_60_row_v2_dual_teacher_api_repilot_before_361",
        "packet_rows": len(packets),
        "exp27a_overlap_rows": len(old_rows),
        "new_risk_focused_rows": len(new_part),
        "new_low_rows": len(low_sel),
        "new_mid_rows": len(mid_sel),
        "new_high_rows": len(high_sel),
        "batch_size": args.batch_size,
        "batch_count": len({packet["batch_id"] for packet in packets}),
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "test_label_read": False,
        "dev_test_used_for_id_guard_only": True,
        **protocol_hashes,
    }
    write_json(out / "decision" / "exp27b_teacher_audit_v2_prepare_decision.json", decision)

    report = [
        "# Exp27B Teacher Audit V2 Re-Pilot Preparation",
        "",
        "This step revises the teacher-audit annotation protocol before scaling. It does not call APIs or train.",
        "",
        "## Counts",
        "",
        f"- v2 re-pilot packets: {len(packets)}",
        f"- Exp27A overlap rows: {len(old_rows)}",
        f"- new risk-focused rows: {len(new_part)}",
        f"- new low rows: {len(low_sel)}",
        f"- new label-3 borderline rows: {len(mid_sel)}",
        f"- new high-control rows: {len(high_sel)}",
        "",
        "## V2 Changes",
        "",
        "- Replaces one overloaded `risk_flag` with `score_region`, `failure_visibility`, and `overestimation_risk`.",
        "- Adds `evidence_type` and `missing_evidence_reason` for missing-content failures.",
        "- Moves answer-key and label-conflict issues out of blind `major_failures`.",
        "- Adds audit-side `label_noise_type`, `recommended_training_use`, and `sample_weight_suggestion`.",
        "",
        "## Guardrails",
        "",
        "- Blind packets do not contain original scores.",
        "- Blind packets do not contain recovered human reasons.",
        "- Dev/test are read only for sample_id/question_key leakage guards.",
        "- Test labels are not read.",
        "- No API call or model training is performed in preparation.",
    ]
    write_text(out / "reports" / "exp27b_teacher_audit_v2_prepare_report.md", "\n".join(report))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Exp27B teacher-audit v2 re-pilot packets.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--exp27a-agreement", type=Path, default=DEFAULT_EXP27A_AGREEMENT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--new-low-count", type=int, default=16)
    parser.add_argument("--new-mid-count", type=int, default=12)
    parser.add_argument("--new-high-count", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
