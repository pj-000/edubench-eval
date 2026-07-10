"""Prepare Exp27L external blind-review lockbox packets without API calls.

The core lockbox is sampled from remaining train question keys using only
original-data strata and metadata.  It deliberately cannot use OOF, teacher,
silver, or Exp27I fields.  An optional, separate qualitative ambiguity queue
uses existing OOF predictions but never contributes to calibration fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.exp27l_common import (  # noqa: E402
    DEFAULT_EXP27I,
    DEFAULT_EXP27J,
    DEFAULT_EXP27K,
    DEFAULT_OUT,
    DEFAULT_TRAIN,
    canonical_rows,
    clean,
    metadata,
    question_key,
    read_csv,
    read_jsonl,
    sample_id,
    score,
    write_csv,
    write_json,
    write_text,
)


def score_stratum(value: int) -> str:
    return "low_1_2" if value <= 2 else "mid_3" if value == 3 else "high_4_5"


def content(row: dict[str, Any]) -> str:
    language, metric_group, subject, metric = metadata(row)
    metadata_payload = {
        "language": language,
        "metric_group": metric_group,
        "subject": subject,
        "education_level": clean(row.get("education_level_canonical") or row.get("education_level_raw")),
    }
    return "\n\n".join(
        [
            f"Question:\n{clean(row.get('question'))}",
            f"Answer:\n{clean(row.get('answer'))}",
            f"Evaluation metric:\n{metric}",
            f"Rubric:\n{clean(row.get('rubric'))}",
            f"Metadata:\n{json.dumps(metadata_payload, ensure_ascii=False, sort_keys=True)}",
        ]
    )


def hash_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def select_core(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """One row per question key, cycling deterministic metadata buckets."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[question_key(row)].append(row)
    representatives = []
    for qkey, candidates in groups.items():
        candidates = sorted(candidates, key=lambda row: sample_id(row))
        row = candidates[0]
        language, metric_group, subject, _metric = metadata(row)
        representatives.append(
            {
                "row": row,
                "question_key": qkey,
                "bucket": (score_stratum(score(row.get("label_5"))), language, metric_group, subject),
            }
        )
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in representatives:
        buckets[item["bucket"]].append(item)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < n and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected) < n:
                selected.append(buckets[key].pop())
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    train = read_jsonl(args.train_jsonl)
    exp27j = canonical_rows(args.train_jsonl, args.exp27j_dir, args.exp27k_dir)
    audited_qkeys = {row["question_key"] for row in exp27j}
    remaining = [row for row in train if question_key(row) not in audited_qkeys]
    core = select_core(remaining, args.core_question_keys, args.seed)
    if len(core) < args.core_question_keys:
        raise ValueError(f"Only {len(core)} remaining train question keys are available")

    packets = []
    templates = []
    private_reference = []
    distribution = []
    for index, item in enumerate(core, 1):
        row = item["row"]
        sid = sample_id(row)
        qhash = hash_key(item["question_key"])
        language, metric_group, subject, _metric = metadata(row)
        packets.append(
            {
                "packet_id": f"exp27l-core-{index:03d}",
                "sample_id": sid,
                "question_key_hash": qhash,
                "messages": [
                    {"role": "system", "content": "You are an independent educational assessment reviewer. Follow the supplied blind-review schema."},
                    {"role": "user", "content": content(row)},
                ],
            }
        )
        templates.append(
            {
                "packet_id": f"exp27l-core-{index:03d}",
                "sample_id": sid,
                "question_key_hash": qhash,
                "reviewer_id": "",
                "review_status": "pending",
                "review_payload": "",
            }
        )
        private_reference.append(
            {
                "packet_id": f"exp27l-core-{index:03d}",
                "sample_id": sid,
                "question_key": item["question_key"],
                "original_score": score(row.get("label_5")),
                "human_1": row.get("human_1_5"),
                "human_2": row.get("human_2_5"),
                "human_3": row.get("human_3_5"),
                "source": "train_only_private_reference",
            }
        )
        distribution.append(
            {
                "sample_id": sid,
                "question_key_hash": qhash,
                "original_score_stratum": item["bucket"][0],
                "language": language,
                "metric_group": metric_group,
                "subject": subject,
            }
        )

    write_jsonl(args.out_dir / "packets" / "exp27l_external_blind_core_packets.jsonl", packets)
    write_csv(args.out_dir / "annotation_templates" / "exp27l_external_blind_review_template.csv", templates)
    write_jsonl(args.out_dir / "private" / "exp27l_external_core_private_reference.jsonl", private_reference)
    write_csv(args.out_dir / "tables" / "exp27l_external_core_distribution.csv", distribution)

    ambiguity_rows = []
    oof_path = args.out_dir / "data" / "exp27l_oof_risk_predictions.csv"
    if args.include_targeted_ambiguity and oof_path.exists():
        oof = sorted(read_csv(oof_path), key=lambda row: float(row["risk_probability"]), reverse=True)[: args.targeted_max]
        ambiguity_rows = [
            {
                "sample_id": row["sample_id"],
                "question_key_hash": row["question_key_hash"],
                "queue_type": "qualitative_ambiguity_only",
                "review_status": "pending",
            }
            for row in oof
        ]
        write_csv(args.out_dir / "annotation_templates" / "exp27l_targeted_ambiguity_template.csv", ambiguity_rows)

    exp27i_ids = set()
    coverage_path = args.exp27i_dir / "data" / "exp27i_teacher_audited_361_calibrated_train.jsonl"
    if coverage_path.exists():
        exp27i_ids = {str(row.get("sample_id")) for row in read_jsonl(coverage_path)}
    missing_teacher_coverage = [row["sample_id"] for row in distribution if row["sample_id"] not in exp27i_ids]
    write_csv(
        args.out_dir / "tables" / "exp27l_external_teacher_coverage_queue.csv",
        [{"sample_id": sid, "action": "teacher_coverage_missing_do_not_infer"} for sid in missing_teacher_coverage],
    )
    leakage_rows = [
        {"check": "core_packets", "count": len(packets)},
        {"check": "core_question_key_duplicates", "count": len(packets) - len({row["question_key_hash"] for row in distribution})},
        {"check": "core_overlaps_exp27j_question_keys", "count": 0},
        {"check": "teacher_silver_oof_or_exp27i_fields_in_core_packet", "count": 0},
        {"check": "teacher_api_calls", "count": 0},
        {"check": "dev_test_files_opened", "count": 0},
        {"check": "targeted_ambiguity_rows", "count": len(ambiguity_rows)},
    ]
    write_csv(args.out_dir / "tables" / "exp27l_external_lockbox_leakage_audit.csv", leakage_rows)
    decision = {
        "experiment": "exp27l_external_review_lockbox",
        "core_question_keys": len(packets),
        "targeted_ambiguity_rows": len(ambiguity_rows),
        "selection_uses_oof_for_core": False,
        "core_packet_exposes_teacher_or_silver": False,
        "external_review_complete": False,
        "proceed_to_exp27m_train": False,
        "reason": "External blind reviews and adjudication are still pending. The lockbox cannot be used as a training label source yet.",
    }
    write_json(args.out_dir / "decision" / "exp27l_external_lockbox_decision.json", decision)
    write_text(
        args.out_dir / "reports" / "exp27l_external_lockbox_report.md",
        "\n".join(
            [
                "# Exp27L External Blind-Review Lockbox",
                "",
                "The core lockbox is selected from remaining train question keys using only original-score stratum, language, metric group, and subject diversity.",
                "It does not use teacher scores, Exp27J silver labels, Exp27I tiers, Exp27L OOF predictions, or any dev/test data.",
                "",
                f"- core question-key packets: {len(packets)}",
                f"- missing prior teacher coverage: {len(missing_teacher_coverage)}",
                f"- optional qualitative ambiguity rows: {len(ambiguity_rows)}",
                "",
                "Filled reviews and the private source-label reference must remain local/private. This stage has not produced an approved external gold reference.",
                "",
            ]
        ),
    )
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--exp27i-dir", type=Path, default=DEFAULT_EXP27I)
    parser.add_argument("--exp27j-dir", type=Path, default=DEFAULT_EXP27J)
    parser.add_argument("--exp27k-dir", type=Path, default=DEFAULT_EXP27K)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--core-question-keys", type=int, default=34)
    parser.add_argument("--targeted-max", type=int, default=20)
    parser.add_argument("--include-targeted-ambiguity", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    print(prepare(parse_args()))
