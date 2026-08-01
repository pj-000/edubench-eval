"""Build train-only manifests for the frozen thesis mechanism controls.

This builder does not read dev/test, load a model, or use a GPU. R3-TOKENAVG
reuses the frozen R3 manifests directly. P1-SYN reuses the frozen matched
synthetic manifest for all three run seeds. Only P1-FULLSEQ needs a new row
artifact: the frozen P1 sequences plus an exact full-completion token mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import read_jsonl, sha256_file
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import reject_eval_path
from thesis_exp.exp54_rar_sft.audit_sorc_dpo_training_manifests import (
    verify_p1_syn_control,
)
from thesis_exp.exp54_rar_sft.training_contract import token_ids_sha256


RAR_ROOT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "mechanism_controls_candidate_v1.json"
)
DEFAULT_CONTRACT = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/THESIS_SCIENTIFIC_CONTRACT_V1.md"
)
DEFAULT_PROMPT_CACHE = RAR_ROOT / "data/shared_prompt_cache.jsonl"
DEFAULT_MATERIALIZED_LOCK = (
    RAR_ROOT / "protocol/materialized_manifest_frozen_lock.json"
)
DEFAULT_PREFERENCE_LOCK = (
    RAR_ROOT
    / "preference_training_candidate/preference_training_frozen_lock.json"
)
DEFAULT_P1 = (
    RAR_ROOT
    / "preference_training_candidate/private/p1_field_dpo.jsonl"
)
DEFAULT_P1_SYN = (
    RAR_ROOT
    / "preference_training_candidate/private/p1_syn_seed42.jsonl"
)
DEFAULT_OUTPUT = RAR_ROOT / "mechanism_controls_candidate"
FULLSEQ_SCHEMA_VERSION = "exp54-p1-fullseq-control-overlay-v1"


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def vector_sha256(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(compact_json(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(compact_json(row) + "\n")
    os.replace(temporary, path)


def attach_full_completion_metadata(
    rows: list[dict[str, Any]],
    prompt_by_record: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy P1 rows and add the exact full assistant-completion masks."""
    output = []
    for row in rows:
        if str(row.get("pair_task")) != "score":
            raise ValueError("P1-FULLSEQ source contains a non-score pair")
        if float(row.get("odpo_offset", -1.0)) != 0.0:
            raise ValueError("P1-FULLSEQ source contains an ODPO offset")
        record_id = str(row.get("record_id") or "")
        prompt = prompt_by_record.get(record_id)
        if prompt is None:
            raise ValueError("P1-FULLSEQ record is absent from prompt cache")
        prompt_ids = list(prompt["prompt_token_ids"])
        if not prompt_ids:
            raise ValueError("prompt cache contains an empty prompt")

        augmented = dict(row)
        augmented["mechanism_control_schema_version"] = FULLSEQ_SCHEMA_VERSION
        for side in ("chosen", "rejected"):
            ids = list(row[f"{side}_input_ids"])
            if ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError(
                    f"{record_id}: {side} sequence has a different prompt prefix"
                )
            if len(ids) <= len(prompt_ids):
                raise ValueError(
                    f"{record_id}: {side} completion is empty"
                )
            completion_positions = list(range(len(prompt_ids), len(ids)))
            field_positions = list(row[f"{side}_field_token_positions"])
            if not set(field_positions).issubset(completion_positions):
                raise ValueError(
                    f"{record_id}: {side} score field escapes completion"
                )
            augmented[f"{side}_prompt_token_count"] = len(prompt_ids)
            augmented[
                f"{side}_completion_token_positions"
            ] = completion_positions
            augmented[
                f"{side}_completion_token_positions_sha256"
            ] = token_ids_sha256(completion_positions)
            augmented[
                f"{side}_completion_token_count"
            ] = len(completion_positions)
        output.append(augmented)
    return output


def _validate_sources(
    *,
    config: dict[str, Any],
    contract_path: Path,
    prompt_cache_path: Path,
    materialized_lock: dict[str, Any],
    preference_lock: dict[str, Any],
    p1_path: Path,
    p1_syn_path: Path,
) -> None:
    if (
        config.get("status")
        != "CPU_IMPLEMENTATION_CANDIDATE_GPU_NOT_AUTHORIZED"
        or config["authorization"]["gpu_smoke_allowed"] is not False
        or config["authorization"]["full_gpu_training_allowed"] is not False
        or config["authorization"]["dev_inference_allowed"] is not False
        or config["authorization"]["test_inference_allowed"] is not False
    ):
        raise PermissionError("mechanism-control candidate is not fail-closed")
    if sha256_file(contract_path) != str(
        config["scientific_contract"]["sha256"]
    ):
        raise ValueError("scientific contract hash differs")
    if (
        materialized_lock.get("status")
        != "MATERIALIZED_MANIFEST_FROZEN_TRAINING_NOT_AUTHORIZED"
        or materialized_lock.get("manifest_frozen") is not True
    ):
        raise ValueError("RAR-SFT materialized manifest lock differs")
    if sha256_file(prompt_cache_path) != str(
        materialized_lock["private_artifact_hashes"]["shared_prompt_cache"]
    ):
        raise ValueError("shared prompt cache differs from frozen RAR-SFT lock")
    if (
        preference_lock.get("status")
        != "SORC_DPO_TRAINING_FROZEN_SMOKE_PACKAGE_BUILD_ALLOWED"
        or preference_lock.get("loss_collator_manifests_and_config_frozen")
        is not True
    ):
        raise ValueError("preference training lock differs")
    expected = preference_lock["private_manifest_hashes"]
    if sha256_file(p1_path) != str(expected["P1_FIELD_DPO"]):
        raise ValueError("frozen P1 manifest differs")
    if sha256_file(p1_syn_path) != str(expected["P1_SYN_SEED42"]):
        raise ValueError("frozen P1-SYN manifest differs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--prompt-cache",
        type=Path,
        default=DEFAULT_PROMPT_CACHE,
    )
    parser.add_argument(
        "--materialized-lock",
        type=Path,
        default=DEFAULT_MATERIALIZED_LOCK,
    )
    parser.add_argument(
        "--preference-lock",
        type=Path,
        default=DEFAULT_PREFERENCE_LOCK,
    )
    parser.add_argument("--p1-manifest", type=Path, default=DEFAULT_P1)
    parser.add_argument(
        "--p1-syn-manifest",
        type=Path,
        default=DEFAULT_P1_SYN,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = (
        args.config,
        args.contract,
        args.prompt_cache,
        args.materialized_lock,
        args.preference_lock,
        args.p1_manifest,
        args.p1_syn_manifest,
    )
    for path in paths:
        reject_eval_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
    output_manifest = args.output_dir / "private/p1_fullseq.jsonl"
    output_report = args.output_dir / "candidate_report.json"
    output_lock = args.output_dir / "candidate_lock.json"
    if any(path.exists() for path in (output_manifest, output_report, output_lock)):
        raise FileExistsError("mechanism-control candidate already exists")

    config = read_json(args.config)
    materialized_lock = read_json(args.materialized_lock)
    preference_lock = read_json(args.preference_lock)
    _validate_sources(
        config=config,
        contract_path=args.contract,
        prompt_cache_path=args.prompt_cache,
        materialized_lock=materialized_lock,
        preference_lock=preference_lock,
        p1_path=args.p1_manifest,
        p1_syn_path=args.p1_syn_manifest,
    )

    prompt_rows = read_jsonl(args.prompt_cache)
    prompt_by_record = {
        str(row["record_id"]): row for row in prompt_rows
    }
    if len(prompt_by_record) != len(prompt_rows):
        raise ValueError("prompt cache contains duplicate record IDs")
    p1_rows = read_jsonl(args.p1_manifest)
    p1_syn_rows = read_jsonl(args.p1_syn_manifest)
    verify_p1_syn_control(p1_rows, p1_syn_rows)
    if len(p1_rows) != 838 or len(p1_syn_rows) != 838:
        raise ValueError("P1/P1-SYN pair count differs from frozen contract")

    fullseq_rows = attach_full_completion_metadata(p1_rows, prompt_by_record)
    write_jsonl(output_manifest, fullseq_rows)
    if len(read_jsonl(output_manifest)) != len(fullseq_rows):
        raise AssertionError("P1-FULLSEQ JSONL round-trip differs")

    r3_hashes = {
        seed: materialized_lock["private_artifact_hashes"][
            "manifests_by_seed"
        ][f"seed{seed}"]["R3"]
        for seed in (42, 43, 44)
    }
    report = {
        "schema_version": "exp54-mechanism-control-candidate-report-v1",
        "status": "MECHANISM_CONTROL_MANIFEST_CANDIDATE_CPU_ONLY",
        "run_count": 9,
        "runs_by_control": {
            "R3_TOKENAVG": 3,
            "P1_FULLSEQ": 3,
            "P1_SYN_LR5E6": 3,
        },
        "p1_fullseq": {
            "pair_count": len(fullseq_rows),
            "source_p1_manifest_sha256": sha256_file(args.p1_manifest),
            "manifest_sha256": sha256_file(output_manifest),
            "sequence_vector_unchanged": True,
            "full_completion_is_contiguous_assistant_suffix": True,
            "chosen_completion_token_total": sum(
                int(row["chosen_completion_token_count"])
                for row in fullseq_rows
            ),
            "rejected_completion_token_total": sum(
                int(row["rejected_completion_token_count"])
                for row in fullseq_rows
            ),
            "chosen_completion_mask_vector_sha256": vector_sha256(
                row["chosen_completion_token_positions_sha256"]
                for row in fullseq_rows
            ),
            "rejected_completion_mask_vector_sha256": vector_sha256(
                row["rejected_completion_token_positions_sha256"]
                for row in fullseq_rows
            ),
        },
        "p1_syn_lr5e6": {
            "pair_count": len(p1_syn_rows),
            "reused_frozen_manifest_sha256": sha256_file(
                args.p1_syn_manifest
            ),
            "same_record_block_chosen_and_field_mask_as_p1": True,
            "run_seeds": [42, 43, 44],
            "old_lr5e7_seed42_checkpoint_reused": False,
        },
        "r3_tokenavg": {
            "reused_frozen_r3_manifest_hashes": r3_hashes,
            "run_seeds": [42, 43, 44],
        },
        "dev_accessed": False,
        "test_accessed": False,
        "gpu_used": False,
        "gpu_smoke_allowed": False,
        "full_gpu_training_allowed": False,
    }
    write_json(output_report, report)
    lock = {
        "schema_version": "exp54-mechanism-control-candidate-lock-v1",
        "status": "MECHANISM_CONTROL_CPU_CANDIDATE_NOT_FROZEN",
        "scientific_contract_sha256": sha256_file(args.contract),
        "config_sha256": sha256_file(args.config),
        "source_hashes": {
            "materialized_manifest_frozen_lock": sha256_file(
                args.materialized_lock
            ),
            "preference_training_frozen_lock": sha256_file(
                args.preference_lock
            ),
            "prompt_cache": sha256_file(args.prompt_cache),
            "p1_manifest": sha256_file(args.p1_manifest),
            "p1_syn_manifest": sha256_file(args.p1_syn_manifest),
        },
        "candidate_hashes": {
            "p1_fullseq_manifest": sha256_file(output_manifest),
            "candidate_report": sha256_file(output_report),
        },
        "gpu_smoke_allowed": False,
        "full_gpu_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "gpu_used": False,
    }
    write_json(output_lock, lock)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
