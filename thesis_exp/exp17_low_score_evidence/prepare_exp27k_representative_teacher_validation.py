"""Prepare missing teacher-audit coverage for the Exp27J representative view.

Exp27K is train-only. It reuses the locked Exp27I target-aware protocol and
creates API packets only for representative Exp27J rows that do not already
have both Qwen and DeepSeek scores. Exp27J silver-reference fields never enter
the teacher packets or the label-aware audit reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import (  # noqa: E402
    read_jsonl,
    sample_id,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp27g_teacher_audit_361_packets import (  # noqa: E402
    label_from_row,
    packet_for,
    question_key,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp27i_target_aware_teacher_audit_361_packets import (  # noqa: E402
    PROMPT_VERSION,
    SCHEMA_VERSION,
    TARGET_VALUE,
    target_aware_packet,
    write_protocol_files,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_TRAIN = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_EXP27I_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27i_target_aware_teacher_audit_361_seed42"
)
DEFAULT_EXP27J_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27j_independent_audit_seed42"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27k_representative_teacher_validation_seed42"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def existing_teacher_coverage(exp27i_dir: Path) -> set[str]:
    path = exp27i_dir / "data" / "exp27i_teacher_audited_361_calibrated_train.jsonl"
    covered: set[str] = set()
    for row in read_jsonl(path):
        if row.get("qwen_score") not in {None, ""} and row.get("deepseek_score") not in {None, ""}:
            covered.add(str(row["sample_id"]))
    return covered


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    required = [
        args.train_jsonl,
        args.exp27j_dir / "tables" / "exp27j_sampling_design.csv",
        args.exp27j_dir / "tables" / "exp27j_adjudicated_manifest.csv",
        args.exp27j_dir / "tables" / "exp27j_leakage_audit.csv",
        args.exp27i_dir / "data" / "exp27i_teacher_audited_361_calibrated_train.jsonl",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    train_rows = read_jsonl(args.train_jsonl)
    train_by_id = {sample_id(row): row for row in train_rows}
    design_rows = read_csv(args.exp27j_dir / "tables" / "exp27j_sampling_design.csv")
    representative_ids = [row["sample_id"] for row in design_rows if row.get("view") == "representative"]
    if len(representative_ids) != 120 or len(set(representative_ids)) != 120:
        raise ValueError(
            f"Expected 120 unique representative rows, got rows={len(representative_ids)} "
            f"unique={len(set(representative_ids))}"
        )
    missing_from_train = sorted(set(representative_ids) - set(train_by_id))
    if missing_from_train:
        raise ValueError(f"Representative ids missing from train: {missing_from_train[:5]}")

    prior_coverage = existing_teacher_coverage(args.exp27i_dir)
    covered_ids = [sid for sid in representative_ids if sid in prior_coverage]
    missing_ids = [sid for sid in representative_ids if sid not in prior_coverage]

    packets: list[dict[str, Any]] = []
    audit_refs: list[dict[str, Any]] = []
    for idx, sid in enumerate(missing_ids):
        row = train_by_id[sid]
        base_packet = packet_for(row, "exp27k_representative_coverage", idx, args.batch_size)
        packet = target_aware_packet(base_packet)
        packets.append(packet)
        audit_refs.append(
            {
                "sample_id": sid,
                "split": "train",
                "pilot_group": "exp27k_representative_coverage",
                "batch_id": packet["batch_id"],
                "original_score": label_from_row(row),
                "human_1": row.get("human_1_5"),
                "human_2": row.get("human_2_5"),
                "human_3": row.get("human_3_5"),
                "human_mean_5": row.get("human_mean_5"),
                "source_meta": packet["source_meta"],
                "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION,
                "target_to_score": TARGET_VALUE,
            }
        )

    out_dir = args.out_dir
    packet_path = out_dir / "packets" / "exp27d_v4_repilot_blind_packets.jsonl"
    ref_path = out_dir / "packets" / "exp27d_v4_repilot_audit_reference_private.jsonl"
    write_jsonl(packet_path, packets)
    write_jsonl(ref_path, audit_refs)
    protocol_hashes = write_protocol_files(out_dir)

    train_ids = set(train_by_id)
    exp27j_leakage = {
        row["check"]: int(row["count"])
        for row in read_csv(args.exp27j_dir / "tables" / "exp27j_leakage_audit.csv")
    }
    inherited_overlap_checks = {
        "dev_sample_overlap": exp27j_leakage.get("dev_sample_overlap", -1),
        "dev_question_overlap": exp27j_leakage.get("dev_question_overlap", -1),
        "test_sample_overlap": exp27j_leakage.get("test_sample_overlap", -1),
        "test_question_overlap": exp27j_leakage.get("test_question_overlap", -1),
    }
    if any(value != 0 for value in inherited_overlap_checks.values()):
        raise ValueError(f"Exp27J leakage guard is not clean: {inherited_overlap_checks}")
    leakage_rows = [
        {"check": "representative_rows", "count": len(representative_ids)},
        {"check": "prior_teacher_covered_rows", "count": len(covered_ids)},
        {"check": "missing_teacher_rows", "count": len(missing_ids)},
        {"check": "missing_rows_in_train", "count": len(set(missing_ids) - train_ids)},
        {"check": "inherited_exp27j_dev_sample_overlap", "count": inherited_overlap_checks["dev_sample_overlap"]},
        {"check": "inherited_exp27j_dev_question_overlap", "count": inherited_overlap_checks["dev_question_overlap"]},
        {"check": "inherited_exp27j_test_sample_overlap", "count": inherited_overlap_checks["test_sample_overlap"]},
        {"check": "inherited_exp27j_test_question_overlap", "count": inherited_overlap_checks["test_question_overlap"]},
        {"check": "silver_reference_fields_in_teacher_packet", "count": 0},
        {"check": "dev_test_files_opened_by_exp27k", "count": 0},
        {"check": "dev_label_used", "count": 0},
        {"check": "test_label_read", "count": 0},
    ]
    write_csv(out_dir / "tables" / "exp27k_coverage_and_leakage.csv", leakage_rows)

    score_counts = Counter(str(row["original_score"]) for row in audit_refs)
    write_csv(
        out_dir / "tables" / "exp27k_missing_representative_distribution.csv",
        [{"original_score": key, "count": value} for key, value in sorted(score_counts.items())],
    )

    hashes = {
        **protocol_hashes,
        "teacher_packet_sha256": file_sha256(packet_path),
        "audit_reference_sha256": file_sha256(ref_path),
    }
    decision = {
        "experiment": "exp27k_representative_teacher_validation",
        "representative_rows": len(representative_ids),
        "prior_teacher_covered_rows": len(covered_ids),
        "missing_teacher_rows": len(missing_ids),
        "expected_full_coverage_after_api": len(covered_ids) + len(missing_ids),
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "target_to_score": TARGET_VALUE,
        "silver_reference_used_in_teacher_prompt": False,
        "human_reason_used_in_teacher_prompt": False,
        "dev_label_used": False,
        "test_label_read": False,
        "dev_test_files_opened": False,
        "api_required": bool(missing_ids),
        "proceed_to_training": False,
        "recommendation": "run_locked_dual_teacher_protocol_then_validate_against_exp27j_silver",
        **hashes,
    }
    write_json(out_dir / "decision" / "exp27k_prepare_decision.json", decision)

    report = [
        "# Exp27K Representative Teacher Validation Preparation",
        "",
        "Exp27K fills missing Qwen/DeepSeek coverage in the Exp27J representative probability sample.",
        "It does not train and does not expose the Exp27J silver reference to either teacher.",
        "",
        "## Coverage",
        "",
        f"- representative rows: {len(representative_ids)}",
        f"- prior teacher-covered rows: {len(covered_ids)}",
        f"- missing rows prepared for API audit: {len(missing_ids)}",
        "",
        "## Protocol",
        "",
        "- blind stage input: question context, evaluator output, metric, rubric, and metadata only.",
        "- label-aware audit stage additionally sees only the original train human score.",
        "- Exp27J reviewer scores, adjudications, failure buckets, and final silver scores are excluded.",
        "- Exp27K does not open dev/test; it inherits Exp27J's already-verified zero-overlap audit.",
        "",
        "## Gate",
        "",
        "Formal downstream training remains blocked until all missing API outputs are complete and protocol validation is rerun.",
    ]
    write_text(out_dir / "reports" / "exp27k_prepare_report.md", "\n".join(report))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Exp27K representative teacher validation packets.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--exp27i-dir", type=Path, default=DEFAULT_EXP27I_DIR)
    parser.add_argument("--exp27j-dir", type=Path, default=DEFAULT_EXP27J_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
