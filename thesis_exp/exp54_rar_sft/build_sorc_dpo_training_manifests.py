"""Materialize private train-only Field-DPO/SORC-DPO candidate manifests.

This builder consumes only the frozen pair artifacts and the already locked
train/tokenizer/R3 checkpoint sources. It never reads dev/test, calls an
evaluator, loads a model, or performs a forward/backward pass.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import (
    load_train_rows,
    read_jsonl,
    sha256_file,
)
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import reject_eval_path
from thesis_exp.exp54_rar_sft.sorc_dpo_loss import (
    exact_objective_weights,
    token_budget,
)
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
MANIFEST_SCHEMA_VERSION = "exp54-sorc-dpo-materialized-pair-v1"


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


def _load_and_verify_pairs(
    pair_lock: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    if (
        pair_lock.get("status")
        != "PAIR_PROTOCOL_FROZEN_IMPLEMENTATION_ALLOWED"
        or pair_lock.get("pair_protocol_frozen") is not True
        or pair_lock.get("training_implementation_allowed") is not True
    ):
        raise PermissionError("pair protocol is not frozen for implementation")
    if (
        pair_lock.get("gpu_preference_training_allowed") is not False
        or pair_lock.get("actual_rationale_preference_labels_created") is not False
        or pair_lock.get("dev_accessed") is not False
        or pair_lock.get("test_accessed") is not False
    ):
        raise PermissionError("pair lock authorization boundary differs")

    expected_hashes = dict(pair_lock["private_artifact_hashes"])
    allowed = set(PAIR_FILES) | {"actual_rationale_candidates"}
    if set(expected_hashes) != allowed:
        raise ValueError("frozen private pair artifact set differs")
    output = {}
    for name, path in PAIR_FILES.items():
        reject_eval_path(path)
        if sha256_file(path) != str(expected_hashes[name]):
            raise ValueError(f"{name}: private pair hash differs from frozen lock")
        rows = read_jsonl(path)
        pair_ids = [str(row.get("pair_hash") or "") for row in rows]
        if any(not value for value in pair_ids):
            raise ValueError(f"{name}: pair_hash is missing")
        if len(set(pair_ids)) != len(pair_ids):
            raise ValueError(f"{name}: duplicate pair_hash")
        output[name] = rows
    return output


def _checkpoint_bindings(
    checkpoint_lock: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if (
        checkpoint_lock.get("status")
        != "SFT_DEV_CHECKPOINT_SELECTION_FROZEN"
        or checkpoint_lock.get("test_accessed") is not False
    ):
        raise ValueError("SFT checkpoint selection is not frozen")
    bindings = {}
    for item in checkpoint_lock["checkpoint_bindings"]:
        if str(item["arm"]).upper() != "R3":
            continue
        seed = str(int(item["seed"]))
        if int(item["selected_epoch"]) != 3:
            raise ValueError("preference cold start must be R3 epoch 3")
        adapter_files = dict(item["adapter_files"])
        bindings[seed] = {
            "seed": int(seed),
            "checkpoint_relative_path": str(item["checkpoint_relative_path"]),
            "adapter_config_sha256": str(
                adapter_files["adapter_config.json"]["sha256"]
            ),
            "adapter_model_sha256": str(
                adapter_files["adapter_model.safetensors"]["sha256"]
            ),
        }
    if set(bindings) != {"42", "43", "44"}:
        raise ValueError("R3 checkpoint lock must bind seeds 42/43/44")
    return bindings


def _full_input_ids(
    prompt: dict[str, Any],
    target: dict[str, Any],
    sequence: dict[str, Any],
) -> list[int]:
    output = (
        list(prompt["prompt_token_ids"])
        + list(target["target_token_ids"])
        + list(sequence["assistant_suffix_token_ids"])
    )
    if len(output) != int(sequence["sequence_token_count"]):
        raise AssertionError("materialized sequence length differs")
    if token_ids_sha256(output) != str(sequence["full_token_ids_sha256"]):
        raise AssertionError("materialized sequence hash differs")
    return output


def _materialize_side(
    tokenizer: Any,
    prompt: dict[str, Any],
    response: dict[str, Any],
    *,
    active_field: str,
    cutoff_len: int,
) -> dict[str, Any]:
    score = int(response["score"])
    rationale = str(response["rationale"])
    target = tokenize_target(
        tokenizer,
        score=score,
        rationale=rationale,
        rationale_active=bool(rationale),
    )
    sequence = materialize_sequence(tokenizer, prompt, target)
    input_ids = _full_input_ids(prompt, target, sequence)
    if len(input_ids) > cutoff_len:
        raise ValueError("preference sequence exceeds cutoff; truncation is forbidden")
    if active_field == "score":
        active_positions = list(sequence["score_token_positions"])
        inactive_positions = list(sequence["rationale_token_positions"])
    elif active_field == "rationale":
        active_positions = list(sequence["rationale_token_positions"])
        inactive_positions = list(sequence["score_token_positions"])
    else:
        raise ValueError(f"unknown active field: {active_field}")
    if not active_positions:
        raise ValueError(f"{active_field} pair has an empty active token mask")
    if set(active_positions) & set(inactive_positions):
        raise AssertionError("active and inactive task fields overlap")
    return {
        "input_ids": input_ids,
        "input_ids_sha256": token_ids_sha256(input_ids),
        "target_bytes_sha256": str(target["target_bytes_sha256"]),
        "target_token_ids_sha256": str(target["target_token_ids_sha256"]),
        "field_token_positions": active_positions,
        "field_token_positions_sha256": token_ids_sha256(active_positions),
        "field_token_count": len(active_positions),
        "inactive_task_field_token_count": len(inactive_positions),
        "sequence_token_count": len(input_ids),
    }


def _materialize_pair(
    tokenizer: Any,
    prompt_by_record: dict[str, dict[str, Any]],
    source: dict[str, Any],
    *,
    pair_task: str,
    offset_mode: str,
    cutoff_len: int,
) -> dict[str, Any]:
    record_id = str(source["record_id"])
    prompt = prompt_by_record.get(record_id)
    if prompt is None:
        raise ValueError("pair record is absent from locked train")
    pair_type = str(source["pair_type"])
    if pair_task == "score":
        if pair_type not in {"adjacent_score", "severe_l2h", "h2l_guard"}:
            raise ValueError("score manifest contains an unknown block")
        if source["chosen"]["rationale"] != source["rejected"]["rationale"]:
            raise ValueError("score pair changes rationale bytes")
        active_field = "score"
    elif pair_task == "rationale":
        if pair_type != "rationale_alignment":
            raise ValueError("rationale manifest contains a non-rationale pair")
        if int(source["chosen"]["score"]) != int(source["rejected"]["score"]):
            raise ValueError("rationale pair changes score")
        active_field = "rationale"
    else:
        raise ValueError("unknown pair task")

    if offset_mode == "zero":
        offset = 0.0
    elif offset_mode == "source":
        offset = float(source["odpo_offset"])
    else:
        raise ValueError("unknown offset mode")
    if pair_task == "rationale" and offset != 0.0:
        raise ValueError("rationale offset must be zero")
    chosen = _materialize_side(
        tokenizer,
        prompt,
        source["chosen"],
        active_field=active_field,
        cutoff_len=cutoff_len,
    )
    rejected = _materialize_side(
        tokenizer,
        prompt,
        source["rejected"],
        active_field=active_field,
        cutoff_len=cutoff_len,
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "pair_id": str(source["pair_hash"]),
        "source_pair_sha256": sha256_bytes(compact_json(source).encode("utf-8")),
        "record_id": record_id,
        "pair_task": pair_task,
        "pair_type": pair_type,
        "pair_source": str(source["pair_source"]),
        "active_field": active_field,
        "odpo_offset": offset,
        "objective_weight": None,
        "chosen_input_ids": chosen["input_ids"],
        "chosen_input_ids_sha256": chosen["input_ids_sha256"],
        "chosen_target_bytes_sha256": chosen["target_bytes_sha256"],
        "chosen_target_token_ids_sha256": chosen["target_token_ids_sha256"],
        "chosen_field_token_positions": chosen["field_token_positions"],
        "chosen_field_token_positions_sha256": (
            chosen["field_token_positions_sha256"]
        ),
        "chosen_field_token_count": chosen["field_token_count"],
        "chosen_inactive_task_field_token_count": (
            chosen["inactive_task_field_token_count"]
        ),
        "rejected_input_ids": rejected["input_ids"],
        "rejected_input_ids_sha256": rejected["input_ids_sha256"],
        "rejected_target_bytes_sha256": rejected["target_bytes_sha256"],
        "rejected_target_token_ids_sha256": rejected["target_token_ids_sha256"],
        "rejected_field_token_positions": rejected["field_token_positions"],
        "rejected_field_token_positions_sha256": (
            rejected["field_token_positions_sha256"]
        ),
        "rejected_field_token_count": rejected["field_token_count"],
        "rejected_inactive_task_field_token_count": (
            rejected["inactive_task_field_token_count"]
        ),
        "cutoff_len": cutoff_len,
    }


def _set_weights(
    rows: list[dict[str, Any]],
    *,
    score_share: float,
    rationale_share: float,
) -> None:
    weights = exact_objective_weights(
        [str(row["pair_task"]) for row in rows],
        [str(row["pair_type"]) for row in rows],
        score_share=score_share,
        rationale_share=rationale_share,
    )
    for row, weight in zip(rows, weights):
        row["objective_weight"] = weight


def _arm_report(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    effective_batch: int,
) -> dict[str, Any]:
    task_counts = Counter(str(row["pair_task"]) for row in rows)
    type_counts = Counter(str(row["pair_type"]) for row in rows)
    weights_by_type = {}
    for pair_type in sorted(type_counts):
        values = {
            float(row["objective_weight"])
            for row in rows
            if row["pair_type"] == pair_type
        }
        if len(values) != 1:
            raise AssertionError("one pair block received multiple objective weights")
        weights_by_type[pair_type] = next(iter(values))
    budget = token_budget(rows)
    return {
        "pair_count": len(rows),
        "pair_task_counts": dict(sorted(task_counts.items())),
        "pair_type_counts": dict(sorted(type_counts.items())),
        "objective_weights_by_pair_type": weights_by_type,
        "objective_weight_sum": sum(
            float(row["objective_weight"]) for row in rows
        ),
        "optimizer_steps_one_physical_pass": math.ceil(
            len(rows) / effective_batch
        ),
        "final_effective_batch_pair_count": (
            len(rows) % effective_batch or effective_batch
        ),
        "token_budget": budget,
        "manifest_sha256": sha256_file(path),
        "pair_id_vector_sha256": vector_sha256(
            str(row["pair_id"]) for row in rows
        ),
        "chosen_sequence_hash_vector_sha256": vector_sha256(
            str(row["chosen_input_ids_sha256"]) for row in rows
        ),
        "rejected_sequence_hash_vector_sha256": vector_sha256(
            str(row["rejected_input_ids_sha256"]) for row in rows
        ),
        "active_field_vector_sha256": vector_sha256(
            str(row["active_field"]) for row in rows
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_report = args.output_dir / "candidate_report.json"
    output_lock = args.output_dir / "candidate_lock.json"
    if output_report.exists() or output_lock.exists():
        raise FileExistsError("preference training candidate already exists")
    for path in (
        args.pair_lock,
        args.training_config,
        args.tokenizer_report,
        args.checkpoint_lock,
        args.reference_lock,
        *PAIR_FILES.values(),
    ):
        reject_eval_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)

    pair_lock = read_json(args.pair_lock)
    config = read_json(args.training_config)
    tokenizer_report = read_json(args.tokenizer_report)
    checkpoint_lock = read_json(args.checkpoint_lock)
    reference_lock = read_json(args.reference_lock)
    if (
        config.get("status")
        != "CPU_IMPLEMENTATION_CANDIDATE_GPU_TRAINING_NOT_ALLOWED"
        or config["authorization"]["gpu_preference_training_allowed"] is not False
        or config["authorization"]["dev_accessed"] is not False
        or config["authorization"]["test_accessed"] is not False
    ):
        raise PermissionError("candidate training config is not fail-closed")
    if (
        reference_lock.get("dev_accessed") is not False
        or reference_lock.get("test_accessed") is not False
    ):
        raise PermissionError("reference lock evaluation boundary differs")
    train_path = (
        REPO_ROOT / "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl"
    )
    reject_eval_path(train_path)
    expected_train_hash = str(
        reference_lock["input"]["rar0_source_hashes"]["train"]
    )
    if sha256_file(train_path) != expected_train_hash:
        raise ValueError("locked train split hash differs")

    pairs = _load_and_verify_pairs(pair_lock)
    train_rows = load_train_rows(train_path)
    train_by_id = {str(row["record_id"]): row for row in train_rows}
    tokenizer = load_locked_tokenizer(args.tokenizer_path, tokenizer_report)
    prompt_by_record = {
        str(row["record_id"]): build_prompt_cache_row(
            tokenizer,
            row,
            row_position=row_position,
        )
        for row_position, row in enumerate(train_rows)
    }
    if set(prompt_by_record) != set(train_by_id):
        raise AssertionError("prompt cache and locked train differ")

    cutoff_len = int(config["data"]["cutoff_len"])
    effective_batch = int(
        config["optimization"]["effective_pair_batch_size"]
    )
    hybrid_zero = [
        _materialize_pair(
            tokenizer,
            prompt_by_record,
            pair,
            pair_task="score",
            offset_mode="zero",
            cutoff_len=cutoff_len,
        )
        for pair in pairs["score_pairs_hybrid"]
    ]
    hybrid_source = [
        _materialize_pair(
            tokenizer,
            prompt_by_record,
            pair,
            pair_task="score",
            offset_mode="source",
            cutoff_len=cutoff_len,
        )
        for pair in pairs["score_pairs_hybrid"]
    ]
    synthetic_zero = [
        _materialize_pair(
            tokenizer,
            prompt_by_record,
            pair,
            pair_task="score",
            offset_mode="zero",
            cutoff_len=cutoff_len,
        )
        for pair in pairs["score_pairs_synthetic_seed42"]
    ]
    rationale = [
        _materialize_pair(
            tokenizer,
            prompt_by_record,
            pair,
            pair_task="rationale",
            offset_mode="zero",
            cutoff_len=cutoff_len,
        )
        for pair in pairs["rationale_pairs_r3_r2"]
    ]
    if vector_sha256(
        row["chosen_input_ids_sha256"] for row in hybrid_zero
    ) != vector_sha256(
        row["chosen_input_ids_sha256"] for row in hybrid_source
    ) or vector_sha256(
        row["rejected_input_ids_sha256"] for row in hybrid_zero
    ) != vector_sha256(
        row["rejected_input_ids_sha256"] for row in hybrid_source
    ):
        raise AssertionError("P1/P2 token materialization differs")

    manifests = {
        "P1_FIELD_DPO": hybrid_zero,
        "P2_SORC_SCORE": hybrid_source,
        "P3_JOINT_SORC": [
            *copy.deepcopy(hybrid_source),
            *rationale,
        ],
        "P1_SYN_SEED42": synthetic_zero,
    }
    shares = {
        "P1_FIELD_DPO": (1.0, 0.0),
        "P2_SORC_SCORE": (1.0, 0.0),
        "P3_JOINT_SORC": (0.5, 0.5),
        "P1_SYN_SEED42": (1.0, 0.0),
    }
    paths = {}
    for arm, rows in manifests.items():
        _set_weights(
            rows,
            score_share=shares[arm][0],
            rationale_share=shares[arm][1],
        )
        path = args.output_dir / "private" / f"{arm.lower()}.jsonl"
        write_jsonl(path, rows)
        paths[arm] = path

    arm_reports = {
        arm: _arm_report(
            rows,
            paths[arm],
            effective_batch=effective_batch,
        )
        for arm, rows in manifests.items()
    }
    for arm, arm_config in config["arms"].items():
        if int(arm_config["pair_count"]) != int(
            arm_reports[arm]["pair_count"]
        ):
            raise ValueError(f"{arm}: candidate pair count differs from config")
        if int(arm_config["optimizer_steps_per_physical_pass"]) != int(
            arm_reports[arm]["optimizer_steps_one_physical_pass"]
        ):
            raise ValueError(f"{arm}: optimizer-step count differs from config")

    report = {
        "schema_version": "exp54-sorc-dpo-training-report-v1",
        "status": "SORC_DPO_TRAINING_MANIFEST_CANDIDATE_NOT_AUTHORIZED",
        "arms": arm_reports,
        "checkpoint_bindings": _checkpoint_bindings(checkpoint_lock),
        "actual_rationale_candidate_count": int(
            pair_lock["aggregate"]["actual_rationale_candidates"]
        ),
        "actual_rationale_candidates_used": False,
        "score_block_equal_weighting_verified": True,
        "p1_p2_pair_and_token_vectors_equal": True,
        "p1_p2_only_offset_differs": True,
        "p3_rationale_qualification_required": True,
        "p3_currently_training_allowed": False,
        "evaluator_called": False,
        "model_loaded": False,
        "forward_backward_executed": False,
        "gpu_preference_training_allowed": False,
        "formal_preference_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    write_json(output_report, report)
    lock = {
        "schema_version": "exp54-sorc-dpo-training-lock-v1",
        "status": "SORC_DPO_TRAINING_CANDIDATE_NOT_FROZEN_NOT_AUTHORIZED",
        "candidate_report_sha256": sha256_file(output_report),
        "private_manifest_hashes": {
            arm: sha256_file(path) for arm, path in paths.items()
        },
        "source_hashes": {
            "pair_protocol_frozen_lock": sha256_file(args.pair_lock),
            "training_config": sha256_file(args.training_config),
            "tokenizer_report": sha256_file(args.tokenizer_report),
            "checkpoint_selection_frozen_lock": sha256_file(
                args.checkpoint_lock
            ),
            "reference_set_data_lock": sha256_file(args.reference_lock),
            "locked_train": sha256_file(train_path),
            "loss_and_collator": sha256_file(
                REPO_ROOT
                / "thesis_exp/exp54_rar_sft/sorc_dpo_loss.py"
            ),
            "manifest_builder": sha256_file(Path(__file__)),
        },
        "pair_artifact_hashes": {
            name: sha256_file(path) for name, path in PAIR_FILES.items()
        },
        "tokenizer_lock_sha256": str(
            tokenizer_report["tokenizer_lock"]["tokenizer_lock_sha256"]
        ),
        "actual_rationale_candidates_used": False,
        "candidate_frozen": False,
        "cpu_unit_tests_required": True,
        "evaluator_calls_allowed": False,
        "gpu_preference_training_allowed": False,
        "formal_preference_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    write_json(output_lock, lock)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
