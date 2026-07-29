"""Independently audit materialized SORC-DPO training candidates.

The auditor derives expected targets and field masks from the frozen source
pairs plus locked train/tokenizer. It does not trust manifest provenance,
invoke the production builder, load a model, call an evaluator, or access
dev/test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import (
    load_train_rows,
    read_jsonl,
    sha256_file,
)
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import reject_eval_path
from thesis_exp.exp54_rar_sft.training_contract import (
    build_prompt_cache_row,
    load_locked_tokenizer,
    materialize_sequence,
    sha256_bytes,
    token_ids_sha256,
    tokenize_target,
)


RAR_ROOT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
PAIR_ROOT = RAR_ROOT / "preference_pairs"
DEFAULT_OUTPUT = RAR_ROOT / "preference_training_candidate"
DEFAULT_PAIR_LOCK = PAIR_ROOT / "pair_protocol_frozen_lock.json"
DEFAULT_TRAINING_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "sorc_dpo_training_candidate_v1.json"
)
DEFAULT_TOKENIZER_REPORT = (
    RAR_ROOT / "audit/r2_event_donor_match_report_seed42.json"
)
DEFAULT_TOKENIZER_PATH = Path(
    "/home/share/models/modelscope/Qwen/Qwen3-4B-Instruct-2507"
)
DEFAULT_CHECKPOINT_LOCK = (
    RAR_ROOT / "protocol/sft_dev_checkpoint_selection_frozen_lock.json"
)
DEFAULT_REFERENCE_LOCK = RAR_ROOT / "protocol/reference_set_data_lock.json"
PAIR_FILES = {
    "score_pairs_hybrid": PAIR_ROOT / "private/score_pairs_hybrid.jsonl",
    "score_pairs_synthetic_seed42": (
        PAIR_ROOT / "private/score_pairs_synthetic_seed42.jsonl"
    ),
    "rationale_pairs_r3_r2": (
        PAIR_ROOT / "private/rationale_pairs_r3_r2.jsonl"
    ),
}
ARM_FILES = {
    "P1_FIELD_DPO": "p1_field_dpo.jsonl",
    "P2_SORC_SCORE": "p2_sorc_score.jsonl",
    "P3_JOINT_SORC": "p3_joint_sorc.jsonl",
    "P1_SYN_SEED42": "p1_syn_seed42.jsonl",
}
ARM_SOURCE = {
    "P1_FIELD_DPO": ("score_pairs_hybrid", "score", "zero"),
    "P2_SORC_SCORE": ("score_pairs_hybrid", "score", "source"),
    "P1_SYN_SEED42": (
        "score_pairs_synthetic_seed42",
        "score",
        "zero",
    ),
}
SCORE_TYPES = ("adjacent_score", "severe_l2h", "h2l_guard")


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def vector_sha256(values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(compact_json(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _expected_side(
    tokenizer: Any,
    prompt: dict[str, Any],
    response: dict[str, Any],
    *,
    active_field: str,
) -> dict[str, Any]:
    rationale = str(response["rationale"])
    target = tokenize_target(
        tokenizer,
        score=int(response["score"]),
        rationale=rationale,
        rationale_active=bool(rationale),
    )
    sequence = materialize_sequence(tokenizer, prompt, target)
    input_ids = (
        list(prompt["prompt_token_ids"])
        + list(target["target_token_ids"])
        + list(sequence["assistant_suffix_token_ids"])
    )
    positions = (
        list(sequence["score_token_positions"])
        if active_field == "score"
        else list(sequence["rationale_token_positions"])
    )
    inactive = (
        list(sequence["rationale_token_positions"])
        if active_field == "score"
        else list(sequence["score_token_positions"])
    )
    return {
        "input_ids": input_ids,
        "input_ids_sha256": token_ids_sha256(input_ids),
        "target_bytes_sha256": str(target["target_bytes_sha256"]),
        "target_token_ids_sha256": str(target["target_token_ids_sha256"]),
        "positions": positions,
        "positions_sha256": token_ids_sha256(positions),
        "inactive_count": len(inactive),
    }


def _expected_weight(
    *,
    arm: str,
    task: str,
    pair_type: str,
    counts: Counter[tuple[str, str]],
    total: int,
) -> float:
    if arm == "P3_JOINT_SORC":
        score_share, rationale_share = 0.5, 0.5
    else:
        score_share, rationale_share = 1.0, 0.0
    if task == "score":
        return total * score_share / (3 * counts[(task, pair_type)])
    if task == "rationale" and rationale_share:
        return total * rationale_share / counts[(task, pair_type)]
    raise ValueError("manifest task is incompatible with arm objective")


def _verify_row(
    row: dict[str, Any],
    source: dict[str, Any],
    prompt: dict[str, Any],
    tokenizer: Any,
    *,
    arm: str,
    task: str,
    offset_mode: str,
    expected_weight: float,
    cutoff_len: int,
) -> None:
    pair_id = str(source["pair_hash"])
    pair_type = str(source["pair_type"])
    active_field = "score" if task == "score" else "rationale"
    expected_offset = (
        float(source["odpo_offset"]) if offset_mode == "source" else 0.0
    )
    fixed = {
        "schema_version": "exp54-sorc-dpo-materialized-pair-v1",
        "pair_id": pair_id,
        "source_pair_sha256": sha256_bytes(
            compact_json(source).encode("utf-8")
        ),
        "record_id": str(source["record_id"]),
        "pair_task": task,
        "pair_type": pair_type,
        "pair_source": str(source["pair_source"]),
        "active_field": active_field,
        "cutoff_len": cutoff_len,
    }
    for field, expected in fixed.items():
        if row.get(field) != expected:
            raise ValueError(f"{arm}/{pair_id}: {field} differs")
    if float(row["odpo_offset"]) != expected_offset:
        raise ValueError(f"{arm}/{pair_id}: ODPO offset differs")
    if not math.isclose(
        float(row["objective_weight"]),
        expected_weight,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{arm}/{pair_id}: objective weight differs")
    if task == "score":
        if source["chosen"]["rationale"] != source["rejected"]["rationale"]:
            raise ValueError(f"{arm}/{pair_id}: score pair changes rationale")
        if pair_type not in SCORE_TYPES:
            raise ValueError(f"{arm}/{pair_id}: unknown score block")
    else:
        if int(source["chosen"]["score"]) != int(source["rejected"]["score"]):
            raise ValueError(f"{arm}/{pair_id}: rationale pair changes score")
        if pair_type != "rationale_alignment" or expected_offset != 0.0:
            raise ValueError(f"{arm}/{pair_id}: rationale contract differs")

    for side in ("chosen", "rejected"):
        expected = _expected_side(
            tokenizer,
            prompt,
            source[side],
            active_field=active_field,
        )
        if len(expected["input_ids"]) > cutoff_len:
            raise ValueError(f"{arm}/{pair_id}: truncation would be required")
        comparisons = {
            f"{side}_input_ids": expected["input_ids"],
            f"{side}_input_ids_sha256": expected["input_ids_sha256"],
            f"{side}_target_bytes_sha256": expected["target_bytes_sha256"],
            f"{side}_target_token_ids_sha256": (
                expected["target_token_ids_sha256"]
            ),
            f"{side}_field_token_positions": expected["positions"],
            f"{side}_field_token_positions_sha256": (
                expected["positions_sha256"]
            ),
            f"{side}_field_token_count": len(expected["positions"]),
            f"{side}_inactive_task_field_token_count": (
                expected["inactive_count"]
            ),
        }
        for field, expected_value in comparisons.items():
            if row.get(field) != expected_value:
                raise ValueError(f"{arm}/{pair_id}: {field} differs")


def _arm_source_rows(
    arm: str,
    pairs: dict[str, list[dict[str, Any]]],
) -> list[tuple[dict[str, Any], str, str]]:
    if arm != "P3_JOINT_SORC":
        source_name, task, offset_mode = ARM_SOURCE[arm]
        return [(row, task, offset_mode) for row in pairs[source_name]]
    return [
        *[
            (row, "score", "source")
            for row in pairs["score_pairs_hybrid"]
        ],
        *[
            (row, "rationale", "zero")
            for row in pairs["rationale_pairs_r3_r2"]
        ],
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pair-lock", type=Path, default=DEFAULT_PAIR_LOCK)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
    )
    parser.add_argument(
        "--tokenizer-report",
        type=Path,
        default=DEFAULT_TOKENIZER_REPORT,
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
    )
    parser.add_argument(
        "--checkpoint-lock",
        type=Path,
        default=DEFAULT_CHECKPOINT_LOCK,
    )
    parser.add_argument(
        "--reference-lock",
        type=Path,
        default=DEFAULT_REFERENCE_LOCK,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_report_path = args.candidate_dir / "candidate_report.json"
    candidate_lock_path = args.candidate_dir / "candidate_lock.json"
    audit_path = args.candidate_dir / "audit_report.json"
    if audit_path.exists():
        raise FileExistsError(audit_path)
    inputs = (
        candidate_report_path,
        candidate_lock_path,
        args.pair_lock,
        args.training_config,
        args.tokenizer_report,
        args.checkpoint_lock,
        args.reference_lock,
        *PAIR_FILES.values(),
    )
    for path in inputs:
        reject_eval_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
    lock = read_json(candidate_lock_path)
    report = read_json(candidate_report_path)
    config = read_json(args.training_config)
    if (
        lock.get("status")
        != "SORC_DPO_TRAINING_CANDIDATE_NOT_FROZEN_NOT_AUTHORIZED"
        or report.get("status")
        != "SORC_DPO_TRAINING_MANIFEST_CANDIDATE_NOT_AUTHORIZED"
    ):
        raise ValueError("candidate state differs")
    if sha256_file(candidate_report_path) != str(
        lock["candidate_report_sha256"]
    ):
        raise ValueError("candidate report hash differs")
    source_paths = {
        "pair_protocol_frozen_lock": args.pair_lock,
        "training_config": args.training_config,
        "tokenizer_report": args.tokenizer_report,
        "checkpoint_selection_frozen_lock": args.checkpoint_lock,
        "reference_set_data_lock": args.reference_lock,
        "locked_train": (
            REPO_ROOT
            / "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl"
        ),
        "loss_and_collator": (
            REPO_ROOT / "thesis_exp/exp54_rar_sft/sorc_dpo_loss.py"
        ),
        "manifest_builder": (
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/"
            "build_sorc_dpo_training_manifests.py"
        ),
    }
    for name, path in source_paths.items():
        if sha256_file(path) != str(lock["source_hashes"][name]):
            raise ValueError(f"candidate source hash differs: {name}")
    for name, path in PAIR_FILES.items():
        if sha256_file(path) != str(lock["pair_artifact_hashes"][name]):
            raise ValueError(f"pair source hash differs: {name}")

    train_rows = load_train_rows(source_paths["locked_train"])
    tokenizer = load_locked_tokenizer(
        args.tokenizer_path,
        read_json(args.tokenizer_report),
    )
    prompts = {
        str(row["record_id"]): build_prompt_cache_row(
            tokenizer,
            row,
            row_position=index,
        )
        for index, row in enumerate(train_rows)
    }
    pairs = {name: read_jsonl(path) for name, path in PAIR_FILES.items()}
    cutoff_len = int(config["data"]["cutoff_len"])
    effective_batch = int(
        config["optimization"]["effective_pair_batch_size"]
    )
    manifest_rows = {}
    aggregate = {}
    for arm, filename in ARM_FILES.items():
        path = args.candidate_dir / "private" / filename
        reject_eval_path(path)
        if sha256_file(path) != str(lock["private_manifest_hashes"][arm]):
            raise ValueError(f"{arm}: manifest hash differs")
        rows = read_jsonl(path)
        source_rows = _arm_source_rows(arm, pairs)
        if len(rows) != len(source_rows):
            raise ValueError(f"{arm}: source and manifest counts differ")
        tasks = [task for _source, task, _offset in source_rows]
        types = [str(source["pair_type"]) for source, _task, _offset in source_rows]
        counts = Counter(zip(tasks, types))
        for row, (source, task, offset_mode) in zip(rows, source_rows):
            record_id = str(source["record_id"])
            _verify_row(
                row,
                source,
                prompts[record_id],
                tokenizer,
                arm=arm,
                task=task,
                offset_mode=offset_mode,
                expected_weight=_expected_weight(
                    arm=arm,
                    task=task,
                    pair_type=str(source["pair_type"]),
                    counts=counts,
                    total=len(rows),
                ),
                cutoff_len=cutoff_len,
            )
        if not math.isclose(
            sum(float(row["objective_weight"]) for row in rows),
            float(len(rows)),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError(f"{arm}: objective weights do not sum to N")
        public_arm = report["arms"][arm]
        expected_public = {
            "pair_count": len(rows),
            "pair_task_counts": dict(
                sorted(Counter(tasks).items())
            ),
            "pair_type_counts": dict(
                sorted(Counter(types).items())
            ),
            "optimizer_steps_one_physical_pass": math.ceil(
                len(rows) / effective_batch
            ),
            "manifest_sha256": sha256_file(path),
            "token_budget": {
                "pairs": len(rows),
                "chosen_unpadded_tokens": sum(
                    len(row["chosen_input_ids"]) for row in rows
                ),
                "rejected_unpadded_tokens": sum(
                    len(row["rejected_input_ids"]) for row in rows
                ),
                "chosen_field_tokens": sum(
                    len(row["chosen_field_token_positions"]) for row in rows
                ),
                "rejected_field_tokens": sum(
                    len(row["rejected_field_token_positions"]) for row in rows
                ),
            },
        }
        for field, expected_value in expected_public.items():
            if public_arm.get(field) != expected_value:
                raise ValueError(f"{arm}: public {field} differs")
        manifest_rows[arm] = rows
        aggregate[arm] = {
            "pairs_verified": len(rows),
            "score_pairs": tasks.count("score"),
            "rationale_pairs": tasks.count("rationale"),
            "optimizer_steps_one_physical_pass": math.ceil(
                len(rows) / effective_batch
            ),
            "semantic_source_mismatches": 0,
            "token_or_mask_mismatches": 0,
            "weight_or_offset_mismatches": 0,
        }

    p1 = manifest_rows["P1_FIELD_DPO"]
    p2 = manifest_rows["P2_SORC_SCORE"]
    ignored = {"odpo_offset"}
    for first, second in zip(p1, p2):
        if {
            key: value for key, value in first.items() if key not in ignored
        } != {
            key: value for key, value in second.items() if key not in ignored
        }:
            raise ValueError("P1/P2 differ outside the ODPO offset")
    p3_score = manifest_rows["P3_JOINT_SORC"][: len(p2)]
    p3_ignored = {"objective_weight"}
    for p2_row, p3_row in zip(p2, p3_score):
        if {
            key: value
            for key, value in p2_row.items()
            if key not in p3_ignored
        } != {
            key: value
            for key, value in p3_row.items()
            if key not in p3_ignored
        }:
            raise ValueError("P3 score block differs from P2")

    for value in (report, lock):
        if (
            value.get("gpu_preference_training_allowed") is not False
            or value.get("formal_preference_training_allowed") is not False
            or value.get("dev_accessed") is not False
            or value.get("test_accessed") is not False
        ):
            raise PermissionError("candidate authorization boundary differs")
    if (
        report.get("actual_rationale_candidates_used") is not False
        or lock.get("actual_rationale_candidates_used") is not False
    ):
        raise PermissionError("unqualified actual rationales entered training")

    audit = {
        "schema_version": "exp54-sorc-dpo-training-audit-v1",
        "status": "SORC_DPO_TRAINING_CANDIDATE_AUDIT_PASS",
        "aggregate": aggregate,
        "p1_p2_only_offset_differs": True,
        "p3_score_block_equals_p2_except_objective_weight": True,
        "three_score_blocks_equal_weighting_verified": True,
        "joint_score_rationale_half_half_verified": True,
        "field_masks_independently_rematerialized": True,
        "actual_rationale_candidates_used": False,
        "candidate_report_sha256": sha256_file(candidate_report_path),
        "candidate_lock_sha256": sha256_file(candidate_lock_path),
        "auditor_source_sha256": sha256_file(Path(__file__)),
        "manifest_hash_vector_sha256": vector_sha256(
            lock["private_manifest_hashes"][arm] for arm in ARM_FILES
        ),
        "evaluator_called": False,
        "model_loaded": False,
        "forward_backward_executed": False,
        "gpu_preference_training_allowed": False,
        "formal_preference_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    write_json(audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
