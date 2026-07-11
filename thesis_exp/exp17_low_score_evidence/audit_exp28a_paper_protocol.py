"""Audit Exp28A inputs against the original paper-like split.

This module performs identity-only overlap checks for dev/test. It never reads
their labels and never constructs training supervision from held-out rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SPLIT_DIR = Path("thesis_exp/data/splits/paper_like_triple_seed42")
DEFAULT_LEGACY_AUDIT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/"
    "exp27i_target_aware_teacher_audit_361_seed42/data/"
    "exp27i_teacher_audited_361_calibrated_train.jsonl"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/"
    "exp28a_paper_protocol_audit_seed42"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def sample_id(row: dict[str, Any]) -> str:
    value = row.get("record_id") or row.get("sample_id")
    if not value:
        raise ValueError("Row is missing record_id/sample_id")
    return str(value)


def question_key(row: dict[str, Any]) -> str:
    value = row.get("question_key") or row.get("question_id")
    if not value:
        raise ValueError(f"Row {sample_id(row)} is missing question_key")
    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    split_paths = {name: args.split_dir / f"{name}.jsonl" for name in ("train", "dev", "test")}
    for path in [*split_paths.values(), args.legacy_audit]:
        if not path.exists():
            raise FileNotFoundError(path)

    # Held-out files are reduced immediately to identity sets. No labels are accessed.
    split_identity: dict[str, dict[str, set[str]]] = {}
    split_counts: dict[str, int] = {}
    for split, path in split_paths.items():
        rows = read_jsonl(path)
        split_counts[split] = len(rows)
        split_identity[split] = {
            "sample_ids": {sample_id(row) for row in rows},
            "question_keys": {question_key(row) for row in rows},
        }

    expected = {"train": 2654, "dev": 664, "test": 2218}
    if split_counts != expected:
        raise ValueError(f"Paper-like split count mismatch: {split_counts} != {expected}")

    all_sample_ids: set[str] = set()
    for split in ("train", "dev", "test"):
        overlap = all_sample_ids & split_identity[split]["sample_ids"]
        if overlap:
            raise ValueError(f"Sample IDs overlap across paper splits: {len(overlap)}")
        all_sample_ids |= split_identity[split]["sample_ids"]

    legacy_rows = read_jsonl(args.legacy_audit)
    legacy_ids = [sample_id(row) for row in legacy_rows]
    if len(legacy_ids) != len(set(legacy_ids)):
        raise ValueError("Legacy audit contains duplicate sample IDs")

    membership_rows: list[dict[str, Any]] = []
    membership = Counter()
    for row in legacy_rows:
        sid = sample_id(row)
        matches = [split for split in ("train", "dev", "test") if sid in split_identity[split]["sample_ids"]]
        split = matches[0] if len(matches) == 1 else "unmatched"
        membership[split] += 1
        membership_rows.append(
            {
                "sample_id": sid,
                "paper_split": split,
                "reuse_status": "eligible_train_reaudit" if split == "train" else "quarantine_heldout",
                "teacher_scores_available": bool(row.get("qwen_score") and row.get("deepseek_score")),
                "model_adjudication_available": bool(row.get("codex_top80_direct_review")),
            }
        )

    overlap_rows: list[dict[str, Any]] = []
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap_rows.append(
            {
                "left_split": left,
                "right_split": right,
                "sample_id_overlap": len(
                    split_identity[left]["sample_ids"] & split_identity[right]["sample_ids"]
                ),
                "question_key_overlap": len(
                    split_identity[left]["question_keys"] & split_identity[right]["question_keys"]
                ),
                "expected_by_protocol": "question overlap allowed; triple/sample overlap forbidden",
            }
        )

    out = args.out_dir
    write_csv(
        out / "tables" / "exp28a_legacy_annotation_membership.csv",
        membership_rows,
        ["sample_id", "paper_split", "reuse_status", "teacher_scores_available", "model_adjudication_available"],
    )
    write_csv(
        out / "tables" / "exp28a_paper_split_overlap.csv",
        overlap_rows,
        ["left_split", "right_split", "sample_id_overlap", "question_key_overlap", "expected_by_protocol"],
    )
    hashes = [
        {"artifact": str(path), "rows": split_counts[name], "sha256": sha256(path)}
        for name, path in split_paths.items()
    ]
    hashes.append(
        {"artifact": str(args.legacy_audit), "rows": len(legacy_rows), "sha256": sha256(args.legacy_audit)}
    )
    write_csv(out / "tables" / "exp28a_source_hashes.csv", hashes, ["artifact", "rows", "sha256"])

    train_reusable = membership["train"]
    heldout_quarantined = membership["dev"] + membership["test"]
    decision = {
        "status": "PASS_WITH_REANNOTATION_REQUIRED",
        "paper_protocol": {
            "train_rows": split_counts["train"],
            "dev_rows": split_counts["dev"],
            "test_rows": split_counts["test"],
            "isolation_unit": "question_answer_metric_triple",
        },
        "legacy_exp27_audit": {
            "total_rows": len(legacy_rows),
            "paper_train_rows": train_reusable,
            "paper_dev_rows_quarantined": membership["dev"],
            "paper_test_rows_quarantined": membership["test"],
            "unmatched_rows": membership["unmatched"],
        },
        "rules": [
            "Only paper-like train rows may contribute teacher labels or adjudications to training.",
            "Legacy paper-dev and paper-test annotations are quarantined and must not guide protocol selection.",
            "Dev is used for model/data-version selection; test is used only after the method is frozen.",
            "Model adjudication must be reported as model_review_silver, never human review.",
        ],
        "next_action": "Build a fresh paper-train-only teacher audit protocol; reuse the 190 legacy train rows only as a separately tagged pilot asset.",
    }
    write_json(out / "decision" / "exp28a_paper_protocol_decision.json", decision)

    report = f"""# Exp28A Paper-Protocol Input Audit

## Decision

**PASS WITH REANNOTATION REQUIRED**

The small-paper campaign is locked to the original paper-like split:

- train: {split_counts['train']}
- dev: {split_counts['dev']}
- test: {split_counts['test']}
- isolation unit: question-answer-metric triple

## Legacy Exp27 Mapping

The existing 361-row teacher audit was selected under `question_seed42`, so it is not a
drop-in training set for the paper protocol:

- {membership['train']} rows map to paper train and may be retained only as tagged pilot evidence.
- {membership['dev']} rows map to paper dev and are quarantined.
- {membership['test']} rows map to paper test and are quarantined.
- {membership['unmatched']} rows are unmatched.

No held-out label was consumed by this audit. Dev/test files were reduced to sample and
question identities solely for overlap checks.

## Locked Rules

1. Teacher annotation and model adjudication may create supervision only for paper train.
2. The 171 legacy held-out annotations cannot be used to choose prompts, fusion rules, or targets.
3. Dev selects the final data version and checkpoint; test remains outside iterative development.
4. Independent model review is reported as model-review silver, not human gold.

## Next Step

Build a fresh paper-train-only annotation inventory and protocol pilot. The 190 reusable legacy
rows remain explicitly tagged so that later ablations can distinguish reused pilot evidence from
new teacher annotations.
"""
    report_path = out / "reports" / "exp28a_paper_protocol_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--legacy-audit", type=Path, default=DEFAULT_LEGACY_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), ensure_ascii=False, sort_keys=True))
