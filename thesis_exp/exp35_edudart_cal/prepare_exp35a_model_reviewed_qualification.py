#!/usr/bin/env python3
"""Prepare Exp35A model-reviewed qualification packets without dev/test access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = Path("thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl")
DEFAULT_EXP33_MANIFEST = Path(
    "thesis_exp/exp33_expert_reference/outputs/exp33a_expert_reference_seed42/"
    "private/exp33a_selected_sample_manifest.jsonl"
)
DEFAULT_OUT = Path(
    "thesis_exp/exp35_edudart_cal/outputs/exp35a_model_reviewed_qualification_seed42"
)
GENERAL_QUOTAS = {3: 20, 4: 40, 5: 60}
SCORE_RUBRIC = {
    "scale": "integer_1_to_5",
    "anchors": {
        "1": "Fails the target dimension in a major or fundamental way.",
        "2": "Substantial target-dimension problems outweigh strengths.",
        "3": "Mixed or adequate performance with material limitations.",
        "4": "Strong performance with only limited target-dimension issues.",
        "5": "Fully satisfies the target dimension with no material failure.",
    },
}


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def guarded(path: Path) -> Path:
    absolute = repo_path(path)
    if absolute.name.casefold() in {"dev.jsonl", "test.jsonl"}:
        raise PermissionError(f"Exp35A forbids access to {absolute.name}")
    return absolute


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with guarded(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text.rstrip() + "\n", encoding="utf-8")


def sid(row: dict[str, Any]) -> str:
    return str(row.get("record_id") or row.get("sample_id") or "")


def qkey(row: dict[str, Any]) -> str:
    return str(row.get("question_key") or "")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def stable_fraction(seed: int, namespace: str, value: str) -> float:
    digest = hashlib.sha256(f"{seed}|{namespace}|{value}".encode()).hexdigest()
    return int(digest[:14], 16) / float(16**14 - 1)


def select_balanced(
    rows: list[dict[str, Any]], quotas: dict[int, int], excluded_ids: set[str], seed: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_qkeys: set[str] = set()
    language_counts: Counter[str] = Counter()
    metric_counts: Counter[str] = Counter()
    for label, quota in sorted(quotas.items()):
        candidates = [row for row in rows if int(row["label_5"]) == label and sid(row) not in excluded_ids]
        if len(candidates) < quota:
            raise ValueError(f"Insufficient label {label}: need {quota}, found {len(candidates)}")
        for _ in range(quota):
            remaining = [row for row in candidates if row not in selected]
            chosen = min(
                remaining,
                key=lambda row: (
                    int(qkey(row) in selected_qkeys),
                    language_counts[str(row.get("language") or "unknown")],
                    metric_counts[str(row.get("metric_group") or "unknown")],
                    stable_fraction(seed, f"qualification-label-{label}", sid(row)),
                ),
            )
            selected.append(chosen)
            selected_qkeys.add(qkey(chosen))
            language_counts[str(chosen.get("language") or "unknown")] += 1
            metric_counts[str(chosen.get("metric_group") or "unknown")] += 1
    return sorted(selected, key=lambda row: (int(row["label_5"]), sid(row)))


def packet_for(row: dict[str, Any], seed: int, view: str) -> dict[str, Any]:
    payload = {
        "sample_id": sid(row),
        "qualification_view": view,
        "anonymized_question_key_hash": hashlib.sha256(
            f"exp35a|{seed}|qkey|{qkey(row)}".encode()
        ).hexdigest(),
        "question_context": (
            "<CONTEXT_ONLY_ORIGINAL_TASK>\n"
            f"{str(row['question']).strip()}\n"
            "</CONTEXT_ONLY_ORIGINAL_TASK>"
        ),
        "evaluator_output": (
            "<EVALUATOR_OUTPUT_TO_SCORE>\n"
            f"{str(row['answer']).strip()}\n"
            "</EVALUATOR_OUTPUT_TO_SCORE>"
        ),
        "evaluation_dimension": row.get("metric_canonical") or row.get("metric_raw"),
        "canonical_rubric": row.get("rubric") or [],
        "score_rubric": SCORE_RUBRIC,
        "non_label_metadata": {
            "subject": row.get("subject_canonical"),
            "education_level": row.get("education_level_canonical"),
            "scenario": row.get("scenario_canonical"),
        },
        "language": row.get("language"),
    }
    payload["packet_hash"] = canonical_hash(payload)
    return payload


def distribution(view: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dimension, field in (
        ("label", "label_5"),
        ("language", "language"),
        ("metric_family", "metric_group"),
    ):
        counts = Counter(str(row.get(field) or "unknown") for row in rows)
        for value, count in sorted(counts.items()):
            matching = [row for row in rows if str(row.get(field) or "unknown") == value]
            output.append(
                {
                    "view": view,
                    "dimension": dimension,
                    "value": value,
                    "rows": count,
                    "share": count / len(rows),
                    "unique_question_keys": len({qkey(row) for row in matching}),
                }
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--exp33-manifest", type=Path, default=DEFAULT_EXP33_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = read_jsonl(args.train)
    if len(train) != 2654:
        raise ValueError(f"Expected locked train=2654, got {len(train)}")
    exp33 = read_jsonl(args.exp33_manifest)
    used_train = {
        str(row["sample_id"])
        for row in exp33
        if row.get("view") in {"representative_train", "risk_enriched_train"}
    }
    low = [row for row in train if int(row["label_5"]) <= 2]
    unused_low = [row for row in low if sid(row) not in used_train]
    if unused_low:
        raise AssertionError("Expected Exp33 to have exhausted train low-tail IDs")

    fresh_general = select_balanced(train, GENERAL_QUOTAS, used_train, args.seed)
    low_reassessment = sorted(low, key=lambda row: (int(row["label_5"]), sid(row)))
    if {sid(row) for row in fresh_general} & used_train:
        raise AssertionError("Fresh general qualification overlaps Exp33 train review IDs")

    views = {
        "fresh_general_qualification": fresh_general,
        "low_tail_reassessment": low_reassessment,
    }
    packets = [
        packet_for(row, args.seed, view)
        for view, rows in views.items()
        for row in rows
    ]
    out = args.out_dir
    for role in ("reviewer_a", "reviewer_b"):
        write_jsonl(out / f"private_review/blind_packets/exp35a_{role}_packet.jsonl", packets)
    manifest = [
        {
            "sample_id": sid(row),
            "qualification_view": view,
            "label_stratum_private": int(row["label_5"]),
            "question_key_private": qkey(row),
            "packet_hash": packet_for(row, args.seed, view)["packet_hash"],
        }
        for view, rows in views.items()
        for row in rows
    ]
    write_jsonl(out / "private/exp35a_selection_manifest.jsonl", manifest)

    audit = {
        "experiment": "Exp35A model-reviewed EduDART-Cal qualification preparation",
        "review_reference": "independent model-reviewed silver, not human expert gold",
        "paper_protocol": "train=2654/dev=664/test=2218",
        "train_rows_read": len(train),
        "dev_rows_read": 0,
        "test_access_count": 0,
        "exp33_train_ids_excluded_from_fresh_general": len(used_train),
        "train_low_rows": len(low),
        "fresh_unused_low_rows": len(unused_low),
        "requested_fresh_low_quota": 40,
        "requested_fresh_low_quota_feasible": False,
        "protocol_adaptation": (
            "Fresh general qualification uses 20 mid and 100 high sample-disjoint rows; "
            "all 76 train-low rows receive a new source-blind reassessment used only as a "
            "low-tail safety stress test because Exp33 exhausted every train-low sample ID."
        ),
        "fresh_general_rows": len(fresh_general),
        "low_tail_reassessment_rows": len(low_reassessment),
        "total_blind_packet_rows": len(packets),
        "fresh_general_exp33_sample_overlap": 0,
        "low_tail_reassessment_is_fresh": False,
        "student_training": False,
        "api_called": False,
        "gpu_used": False,
    }
    write_json(out / "decision/exp35a_preparation_decision.json", audit)
    distribution_rows = [row for view, rows in views.items() for row in distribution(view, rows)]
    write_csv(
        out / "tables/exp35a_qualification_distribution.csv",
        distribution_rows,
        ["view", "dimension", "value", "rows", "share", "unique_question_keys"],
    )
    write_text(
        out / "reports/exp35a_preparation_report.md",
        f"""# Exp35A Model-Reviewed Qualification Preparation

- Reference status: independent model-reviewed silver; not human expert gold.
- Locked train read: 2,654 rows. Dev/test read: 0/0.
- Fresh general qualification: 120 rows (label3=20, label4=40, label5=60).
- Low-tail reassessment: all {len(low_reassessment)} train low rows, reviewed in new blind runs.
- Fresh low quota requested: 40; feasible unused train-low rows: 0.
- Reason: Exp33 representative/risk views already used all 76 train-low IDs.
- Adaptation: low-tail reassessment is explicitly a repeated-sample safety stress test,
  never an independent fresh qualification view or prevalence estimate.
- Packet fields contain no original human score, Qwen, DeepSeek, Exp33 silver, or student prediction.
- Training/inference/API/GPU: none.
""",
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
