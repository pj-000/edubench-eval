"""Resolve and lock existing Exp47A inputs without opening dev or test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_exp.exp47_label2_identifiability.common import (
    E4_PATH,
    EXPECTED_FOLDS,
    EXPECTED_ROWS,
    EXP44_RUN_ROOT,
    EXP46_PUBLIC_ROOT,
    EXP46_RUN_ROOT,
    FOLD_PATH,
    ROOT,
    TRAIN_PATH,
    ensure_dirs,
    input_hashes,
    sha256_file,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--exp44-run-root", type=Path, default=EXP44_RUN_ROOT)
    parser.add_argument("--exp46-run-root", type=Path, default=EXP46_RUN_ROOT)
    return parser.parse_args()


def artifact_row(name: str, path: Path, expected_rows: int | None = None) -> dict[str, object]:
    exists = path.exists()
    size = path.stat().st_size if exists and path.is_file() else None
    hash_allowed = size is not None and size <= 50 * 1024 * 1024
    return {
        "artifact": name,
        "path": str(path),
        "exists": exists,
        "expected_rows": expected_rows if expected_rows is not None else "",
        "availability": "AVAILABLE" if exists else "MISSING",
        "byte_size": size if size is not None else "",
        "sha256": sha256_file(path) if hash_allowed else "",
        "hash_status": "HASHED" if hash_allowed else "SKIPPED_HEAVY_PRIVATE" if size is not None else "NOT_AVAILABLE",
    }


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    artifacts = [
        artifact_row("paper_like_train", TRAIN_PATH, EXPECTED_ROWS),
        artifact_row("exp43_e4_private_optional", E4_PATH, EXPECTED_ROWS),
        artifact_row("fold_assignment", FOLD_PATH, EXPECTED_ROWS),
        artifact_row("exp46_teacher_decision", EXP46_PUBLIC_ROOT / "decision/exp46a_teacher_capacity_decision.json"),
    ]
    for fold in EXPECTED_FOLDS:
        base06 = args.exp44_run_root / f"groupcv/C0_E4_baseline/seed_42/fold_{fold}"
        base4 = args.exp46_run_root / f"groupcv/T1_4B_teacher/seed_42/fold_{fold}"
        artifacts.extend(
            [
                artifact_row(f"0.6b_fold_{fold}_heldout", base06 / "heldout_predictions.jsonl"),
                artifact_row(f"0.6b_fold_{fold}_summary", base06 / "run_summary.json"),
                artifact_row(f"0.6b_fold_{fold}_checkpoint", base06 / "model.pt"),
                artifact_row(f"4b_fold_{fold}_all_logits", base4 / "teacher_logits_all.jsonl", EXPECTED_ROWS),
                artifact_row(f"4b_fold_{fold}_heldout", base4 / "heldout_predictions.jsonl"),
                artifact_row(f"4b_fold_{fold}_summary", base4 / "run_summary.json"),
                artifact_row(f"4b_fold_{fold}_adapter", base4 / "teacher_adapter_and_head.pt"),
            ]
        )
    optional_names = {"exp43_e4_private_optional"}
    required = [
        row
        for row in artifacts
        if row["artifact"] not in optional_names
        and not str(row["artifact"]).endswith("_checkpoint")
        and not str(row["artifact"]).endswith("_adapter")
    ]
    missing_required = [row["artifact"] for row in required if not row["exists"]]
    if missing_required:
        raise FileNotFoundError(f"Missing required Exp47 inputs: {missing_required}")

    write_csv(args.out_dir / "tables/exp47_input_resolution.csv", artifacts)
    protocol = {
        "experiment": "Exp47A Label-2 Identifiability and Generalization Audit",
        "locked_source_commit": "9b1e77e",
        "no_training": True,
        "no_api": True,
        "dev_access_allowed": False,
        "test_access_allowed": False,
        "model_0.6b": {
            "outer_train_status": "MISSING_CHECKPOINT_NOT_RECOMPUTED",
            "heldout_status": "EXISTING_OOF_ONLY",
        },
        "model_4b": {
            "outer_train_status": "EXISTING_LOGITS",
            "heldout_status": "EXISTING_LOGITS",
            "new_inference_required": False,
        },
        "mutually_exclusive_label2_subtypes": {
            "strict_222": "sorted human scores equal [2,2,2]",
            "stable_majority_non_strict": "at least two scores equal 2, range < 3, excluding strict_222",
            "ambiguous": "all remaining hard-label2 rows, including fewer than two scores of 2 or range >= 3",
        },
        "primary_decision_precedence": [
            "LABEL2_TARGET_AMBIGUOUS",
            "QUESTION_GENERALIZATION_LIMIT",
            "OBJECTIVE_OR_ADAPTATION_LIMIT",
            "MIXED_DATA_AND_ADAPTATION_LIMIT",
        ],
        "secondary_diagnosis_flags_required": True,
    }
    write_json(args.out_dir / "configs/exp47a_audit_protocol_lock.json", protocol)
    write_json(
        args.out_dir / "hashes/exp47a_input_hashes.json",
        {
            "public_input_hashes": input_hashes(),
            "private_artifact_hashes": {
                str(row["artifact"]): row["sha256"]
                for row in artifacts
                if row["hash_status"] == "HASHED"
            },
            "resolved_artifacts": len(artifacts),
            "missing_0.6b_outer_train_checkpoint": True,
            "dev_access_count": 0,
            "test_access_count": 0,
        },
    )
    print(json.dumps({"status": "INPUTS_RESOLVED", "new_inference_required": False, "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
