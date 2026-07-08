"""Prepare Exp27A teacher-audit pilot packets.

This step does not call any model. It builds a deterministic pilot set and
annotation packet files for a two-stage teacher-audited EduBench protocol:

1. blind rubric scoring, where the teacher sees no original label;
2. label audit, where the teacher sees the original label only after producing
   a blind judgment.

The output is intentionally train-focused. Dev/test are not used by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
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
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27a_teacher_audit_pilot_seed42")

SCHEMA_VERSION = "exp27a_teacher_audit_v1"
PROMPT_VERSION = "exp27a_v1"

FAILURE_TAGS = [
    "missing_key_point",
    "factual_or_rubric_mismatch",
    "answer_key_or_reference_mismatch",
    "surface_fluent_but_hidden_defect",
    "insufficient_evidence",
    "partial_or_incomplete",
    "task_constraint_violation",
    "format_violation",
    "possible_label_conflict",
    "no_major_failure",
    "unclear",
]
RISK_FLAGS = ["hidden_low_failure", "borderline", "clean_high", "unclear"]
CONFIDENCE_VALUES = ["high", "medium", "low"]
AGREEMENT_VALUES = ["exact", "adjacent", "conflict", "unclear"]
QUALITY_VALUES = ["reliable", "plausible_adjacent", "suspected_conflict", "unclear"]


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def preview(value: Any, limit: int = 240) -> str:
    return re.sub(r"\s+", " ", clean(value)).strip()[:limit]


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


def row_signature(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (str(clamp_score(row.get("label_5"))), language(row), metric_name(row), subject(row))


def stratified_take(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if n <= 0 or not rows:
        return []
    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(row_signature(row), []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    keys = list(buckets)
    rng.shuffle(keys)
    while len(selected) < n and keys:
        next_keys: list[tuple[str, str, str, str]] = []
        for key in keys:
            bucket = buckets[key]
            if bucket and len(selected) < n:
                selected.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected[:n]


def user_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": clean(row.get("question")),
        "answer": clean(row.get("answer")),
        "metric": metric_name(row),
        "rubric": clean(row.get("rubric")),
        "metadata": teacher_metadata_text(row),
    }


def packet_for(row: dict[str, Any], group: str, batch_size: int) -> dict[str, Any]:
    sid = sample_id(row)
    batch_index = int(packet_for.counter // batch_size) + 1
    packet_for.counter += 1
    return {
        "sample_id": sid,
        "split": "train",
        "pilot_group": group,
        "batch_id": f"exp27a_pilot_{batch_index:03d}",
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "teacher_input": user_payload(row),
        "source_meta": {
            "question_key": question_key(row),
            "language": language(row),
            "subject": subject(row),
            "metric": metric_name(row),
            "sample_hash": sha1(json.dumps(user_payload(row), ensure_ascii=False, sort_keys=True)),
        },
    }


packet_for.counter = 0  # type: ignore[attr-defined]


def make_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Exp27A Teacher Audit Annotation",
        "type": "object",
        "additionalProperties": False,
        "required": ["sample_id", "blind", "audit"],
        "properties": {
            "sample_id": {"type": "string"},
            "blind": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "teacher_score",
                    "teacher_reason",
                    "major_failures",
                    "evidence_span",
                    "rubric_clause",
                    "score_cap",
                    "risk_flag",
                    "confidence",
                ],
                "properties": {
                    "teacher_score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "teacher_reason": {"type": "string"},
                    "major_failures": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "enum": FAILURE_TAGS},
                    },
                    "evidence_span": {"type": ["string", "null"]},
                    "rubric_clause": {"type": ["string", "null"]},
                    "score_cap": {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
                    "risk_flag": {"type": "string", "enum": RISK_FLAGS},
                    "confidence": {"type": "string", "enum": CONFIDENCE_VALUES},
                },
            },
            "audit": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "original_score",
                    "score_agreement",
                    "label_quality",
                    "needs_human_review",
                    "audit_reason",
                ],
                "properties": {
                    "original_score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "score_agreement": {"type": "string", "enum": AGREEMENT_VALUES},
                    "label_quality": {"type": "string", "enum": QUALITY_VALUES},
                    "needs_human_review": {"type": "boolean"},
                    "audit_reason": {"type": "string"},
                },
            },
        },
    }


def blind_prompt() -> str:
    tags = ", ".join(FAILURE_TAGS)
    return f"""You are a strict educational assessment teacher-auditor.

Task: assign a 1-5 score for ONE answer using only the given question, answer,
metric, rubric, and metadata. You must not assume any original human score.

Return exactly one JSON object matching the provided schema. Do not wrap the
JSON in Markdown. Do not include hidden reasoning or chain-of-thought.

Rules:
- Ground the score in the rubric, not in surface fluency.
- If the answer deserves a low score, identify the concrete rubric-linked
  failure and quote an exact evidence_span from the answer when possible.
- evidence_span must be a substring of the answer. If no localizable span exists,
  use null and set confidence no higher than medium.
- rubric_clause must quote or closely match the relevant rubric clause.
- score_cap is the maximum reasonable score if a serious failure exists; use
  null when no cap is needed.
- major_failures must use only these tags: {tags}.
- Use no_major_failure only when no material rubric-linked failure is evident.
- Do not penalize length, style, or wording unless the rubric requires it.
"""


def audit_prompt() -> str:
    return """You are auditing whether the original human score is reliable.

You will receive:
1. the original question/answer/rubric input,
2. your previous blind scoring result,
3. the original human score.

Return exactly one JSON object matching the provided schema. Keep the blind
fields unchanged unless they are invalid JSON or violate the schema. In audit,
decide whether the original human score is reliable, adjacent/plausible, in
conflict with the blind judgment, or unclear.

Do not treat the teacher score as gold. The teacher is an auditor, not a
replacement for human labels.
"""


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_ids(path: Path) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    qkeys: set[str] = set()
    for row in read_jsonl(path):
        ids.add(sample_id(row))
        qkeys.add(question_key(row))
    return ids, qkeys


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    rows = read_jsonl(args.train_jsonl)
    low = [row for row in rows if clamp_score(row.get("label_5")) <= 2]
    mid = [row for row in rows if clamp_score(row.get("label_5")) == 3]
    high = [row for row in rows if clamp_score(row.get("label_5")) >= 4]
    mid_sel = stratified_take(mid, args.mid_count, args.seed + 11)
    high_sel = stratified_take(high, args.high_count, args.seed + 29)
    selected: list[tuple[str, dict[str, Any]]] = [("train_low_all", row) for row in low]
    selected.extend(("train_mid_stratified", row) for row in mid_sel)
    selected.extend(("train_high_control_stratified", row) for row in high_sel)
    rng.shuffle(selected)
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
    write_jsonl(out / "packets" / "exp27a_pilot_blind_packets.jsonl", packets)
    write_jsonl(out / "packets" / "exp27a_pilot_audit_reference_private.jsonl", audit_reference)
    write_text(out / "prompts" / "exp27a_blind_teacher_prompt.md", blind_prompt())
    write_text(out / "prompts" / "exp27a_label_audit_prompt.md", audit_prompt())
    write_json(out / "schema" / "exp27a_teacher_audit_schema.json", make_schema())

    ref_by_sid = {row["sample_id"]: row for row in audit_reference}
    distribution = Counter((p["pilot_group"], ref_by_sid[p["sample_id"]]["original_score"]) for p in packets)
    write_csv(
        out / "tables" / "exp27a_pilot_sample_distribution.csv",
        [
            {"pilot_group": group, "label": label, "count": count}
            for (group, label), count in sorted(distribution.items())
        ],
    )
    batch_counts = Counter(p["batch_id"] for p in packets)
    write_csv(
        out / "tables" / "exp27a_batch_manifest.csv",
        [{"batch_id": key, "count": value} for key, value in sorted(batch_counts.items())],
    )

    dev_ids, dev_qkeys = load_ids(args.dev_jsonl)
    test_ids, test_qkeys = load_ids(args.test_jsonl)
    packet_ids = {p["sample_id"] for p in packets}
    packet_qkeys = {p["source_meta"]["question_key"] for p in packets}
    leakage_rows = [
        {"check": "dev_sample_overlap", "count": len(packet_ids & dev_ids)},
        {"check": "dev_question_overlap", "count": len(packet_qkeys & dev_qkeys)},
        {"check": "test_sample_overlap", "count": len(packet_ids & test_ids)},
        {"check": "test_question_overlap", "count": len(packet_qkeys & test_qkeys)},
        {"check": "test_label_read", "count": 0},
    ]
    write_csv(out / "tables" / "exp27a_packet_leakage_audit.csv", leakage_rows)

    decision = {
        "recommendation": "run_dual_teacher_pilot_before_full_5536_annotation",
        "pilot_rows": len(packets),
        "train_low_rows": len(low),
        "mid_rows": len(mid_sel),
        "high_control_rows": len(high_sel),
        "batch_size": args.batch_size,
        "batch_count": len(batch_counts),
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "test_label_read": False,
        "dev_test_used_for_id_guard_only": True,
    }
    write_json(out / "decision" / "exp27a_teacher_audit_pilot_decision.json", decision)
    report = [
        "# Exp27A Teacher Audit Pilot Packets",
        "",
        "This step prepares a two-stage teacher-audited annotation pilot. It does not call any model.",
        "",
        "## Counts",
        "",
        f"- pilot rows: {len(packets)}",
        f"- train low rows: {len(low)}",
        f"- train mid sampled rows: {len(mid_sel)}",
        f"- train high-control sampled rows: {len(high_sel)}",
        f"- batches: {len(batch_counts)}",
        "",
        "## Guardrails",
        "",
        "- Blind packet prompts do not expose the original score to the teacher.",
        "- Original scores for audit are stored only in `packets/exp27a_pilot_audit_reference_private.jsonl`.",
        "- Dev/test are read only for sample_id/question_key leakage guards.",
        "- Test labels are not read.",
        "- Teacher output is an audit signal, not a replacement gold label.",
    ]
    write_text(out / "reports" / "exp27a_teacher_audit_pilot_report.md", "\n".join(report))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Exp27A teacher-audit pilot packets.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mid-count", type=int, default=100)
    parser.add_argument("--high-count", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
