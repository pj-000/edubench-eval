"""Independently audit the train-only mechanism-control candidate manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import read_jsonl, sha256_file
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import reject_eval_path
from thesis_exp.exp54_rar_sft.training_contract import token_ids_sha256


RAR_ROOT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
DEFAULT_CANDIDATE = RAR_ROOT / "mechanism_controls_candidate"
DEFAULT_PROMPT_CACHE = RAR_ROOT / "data/shared_prompt_cache.jsonl"
DEFAULT_P1 = (
    RAR_ROOT
    / "preference_training_candidate/private/p1_field_dpo.jsonl"
)
DEFAULT_P1_SYN = (
    RAR_ROOT
    / "preference_training_candidate/private/p1_syn_seed42.jsonl"
)
ALLOWED_OVERLAY_KEYS = {
    "mechanism_control_schema_version",
    "chosen_prompt_token_count",
    "chosen_completion_token_positions",
    "chosen_completion_token_positions_sha256",
    "chosen_completion_token_count",
    "rejected_prompt_token_count",
    "rejected_completion_token_positions",
    "rejected_completion_token_positions_sha256",
    "rejected_completion_token_count",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _audit_fullseq_rows(
    source_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    prompt_by_record: dict[str, dict[str, Any]],
) -> None:
    if len(source_rows) != 838 or len(control_rows) != 838:
        raise ValueError("P1/P1-FULLSEQ must each contain 838 pairs")
    for index, (source, control) in enumerate(
        zip(source_rows, control_rows, strict=True)
    ):
        if set(control) - set(source) != ALLOWED_OVERLAY_KEYS:
            raise ValueError(f"row {index}: overlay field set differs")
        for key, value in source.items():
            if control.get(key) != value:
                raise ValueError(f"row {index}: frozen P1 field changed: {key}")
        if (
            control["mechanism_control_schema_version"]
            != "exp54-p1-fullseq-control-overlay-v1"
        ):
            raise ValueError(f"row {index}: overlay schema differs")
        record_id = str(source["record_id"])
        prompt = prompt_by_record.get(record_id)
        if prompt is None:
            raise ValueError(f"row {index}: prompt record is missing")
        prompt_ids = list(prompt["prompt_token_ids"])
        for side in ("chosen", "rejected"):
            ids = list(source[f"{side}_input_ids"])
            prompt_count = int(control[f"{side}_prompt_token_count"])
            positions = list(
                control[f"{side}_completion_token_positions"]
            )
            expected = list(range(len(prompt_ids), len(ids)))
            if prompt_count != len(prompt_ids) or ids[:prompt_count] != prompt_ids:
                raise ValueError(f"row {index}: {side} prompt binding differs")
            if positions != expected:
                raise ValueError(f"row {index}: {side} completion mask differs")
            if (
                token_ids_sha256(positions)
                != control[f"{side}_completion_token_positions_sha256"]
                or len(positions)
                != int(control[f"{side}_completion_token_count"])
            ):
                raise ValueError(f"row {index}: {side} completion lock differs")
            if not set(source[f"{side}_field_token_positions"]).issubset(
                positions
            ):
                raise ValueError(f"row {index}: {side} score mask escapes completion")


def _audit_p1_syn_match(
    p1_rows: list[dict[str, Any]],
    p1_syn_rows: list[dict[str, Any]],
) -> None:
    if len(p1_rows) != 838 or len(p1_syn_rows) != 838:
        raise ValueError("P1/P1-SYN must each contain 838 pairs")
    equal_keys = (
        "record_id",
        "pair_type",
        "pair_task",
        "active_field",
        "chosen_input_ids",
        "chosen_field_token_positions",
        "objective_weight",
        "cutoff_len",
    )
    for index, (actual, synthetic) in enumerate(
        zip(p1_rows, p1_syn_rows, strict=True)
    ):
        for key in equal_keys:
            if actual[key] != synthetic[key]:
                raise ValueError(
                    f"row {index}: P1/P1-SYN matched field changed: {key}"
                )
        if float(actual["odpo_offset"]) != 0.0 or float(
            synthetic["odpo_offset"]
        ) != 0.0:
            raise ValueError("P1/P1-SYN must both use zero offset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=DEFAULT_CANDIDATE,
    )
    parser.add_argument(
        "--prompt-cache",
        type=Path,
        default=DEFAULT_PROMPT_CACHE,
    )
    parser.add_argument("--p1-manifest", type=Path, default=DEFAULT_P1)
    parser.add_argument(
        "--p1-syn-manifest",
        type=Path,
        default=DEFAULT_P1_SYN,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fullseq_path = args.candidate_dir / "private/p1_fullseq.jsonl"
    report_path = args.candidate_dir / "candidate_report.json"
    lock_path = args.candidate_dir / "candidate_lock.json"
    audit_path = args.candidate_dir / "audit_report.json"
    for path in (
        fullseq_path,
        report_path,
        lock_path,
        args.prompt_cache,
        args.p1_manifest,
        args.p1_syn_manifest,
    ):
        reject_eval_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
    if audit_path.exists():
        raise FileExistsError(audit_path)

    source_rows = read_jsonl(args.p1_manifest)
    synthetic_rows = read_jsonl(args.p1_syn_manifest)
    control_rows = read_jsonl(fullseq_path)
    prompt_rows = read_jsonl(args.prompt_cache)
    prompt_by_record = {
        str(row["record_id"]): row for row in prompt_rows
    }
    if len(prompt_by_record) != len(prompt_rows):
        raise ValueError("prompt cache contains duplicate record IDs")
    _audit_fullseq_rows(source_rows, control_rows, prompt_by_record)
    _audit_p1_syn_match(source_rows, synthetic_rows)

    candidate_report = read_json(report_path)
    candidate_lock = read_json(lock_path)
    if (
        candidate_report.get("status")
        != "MECHANISM_CONTROL_MANIFEST_CANDIDATE_CPU_ONLY"
        or int(candidate_report.get("run_count", -1)) != 9
        or candidate_report.get("dev_accessed") is not False
        or candidate_report.get("test_accessed") is not False
        or candidate_lock.get("status")
        != "MECHANISM_CONTROL_CPU_CANDIDATE_NOT_FROZEN"
        or candidate_lock["candidate_hashes"]["p1_fullseq_manifest"]
        != sha256_file(fullseq_path)
        or candidate_lock["candidate_hashes"]["candidate_report"]
        != sha256_file(report_path)
    ):
        raise ValueError("candidate report/lock binding differs")

    audit = {
        "schema_version": "exp54-mechanism-control-audit-report-v1",
        "status": "MECHANISM_CONTROL_CPU_AUDIT_PASS_GPU_NOT_AUTHORIZED",
        "p1_fullseq_rows_verified": len(control_rows),
        "p1_fullseq_source_fields_byte_equivalent": True,
        "full_completion_masks_rederived_from_prompt_cache": True,
        "p1_p1_syn_same_record_block_chosen_mask_and_weight": True,
        "candidate_report_sha256": sha256_file(report_path),
        "candidate_lock_sha256": sha256_file(lock_path),
        "p1_fullseq_manifest_sha256": sha256_file(fullseq_path),
        "gpu_used": False,
        "gpu_smoke_allowed": False,
        "full_gpu_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    write_json(audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
