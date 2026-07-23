"""Audit private candidate materialized manifests and publish aggregate locks.

No human rationale, row identifier, target text, token list, or row-level
mapping is written to the public report or lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import (
    file_sha256,
    read_jsonl,
    reject_eval_path,
    write_json,
)
from thesis_exp.exp54_rar_sft.build_training_manifests import (
    ARMS,
    DEFAULT_OUTPUT,
    EXPECTED_EVENTS_PER_SEED,
)
from thesis_exp.exp54_rar_sft.reference_schedule import (
    EXPECTED_TRAIN_ROWS,
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
)
from thesis_exp.exp54_rar_sft.training_contract import token_ids_sha256


DEFAULT_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/materialized_manifest_candidate.json"
)
SOURCE_PATHS = {
    "training_contract": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/training_contract.py"
    ),
    "manifest_builder": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/build_training_manifests.py"
    ),
    "manifest_auditor": Path(__file__),
    "block_loss_and_collator": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/block_loss.py"
    ),
    "output_schema": (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/schemas/rar_v2_output_schema.json"
    ),
}
UPSTREAM_LOCK_PATHS = {
    "reference_set_lock": (
        DEFAULT_OUTPUT / "protocol/reference_set_data_lock.json"
    ),
    "base_schedule_lock": (
        DEFAULT_OUTPUT / "protocol/reference_schedule_candidate_lock.json"
    ),
    "event_donor_lock": (
        DEFAULT_OUTPUT / "protocol/r2_event_donor_map_candidate_lock.json"
    ),
    "event_mask_lock": (
        DEFAULT_OUTPUT / "protocol/r2_r3_event_mask_candidate_lock.json"
    ),
    "tokenizer_report_seed42": (
        DEFAULT_OUTPUT / "audit/r2_event_donor_match_report_seed42.json"
    ),
}


def _vector_sha256(values: Iterable[Any]) -> str:
    payload = json.dumps(
        list(values),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _counter_sha256(counter: Counter[str]) -> str:
    return _vector_sha256(sorted(counter.items()))


def _counter_l1(left: Counter[str], right: Counter[str]) -> int:
    return sum(
        abs(left[key] - right[key])
        for key in set(left) | set(right)
    )


def _require_exact_manifest(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    seed: int,
    config: dict[str, Any],
) -> None:
    if len(rows) != EXPECTED_EVENTS_PER_SEED:
        raise ValueError(f"{arm}/seed {seed}: wrong manifest row count")
    if len({str(row["base_event_id"]) for row in rows}) != len(rows):
        raise ValueError(f"{arm}/seed {seed}: duplicate base event ID")
    expected_cutoff = int(config["cutoff_len"])
    for index, row in enumerate(rows):
        epoch_index, row_position = divmod(index, EXPECTED_TRAIN_ROWS)
        expected = {
            "candidate_status": "CANDIDATE_NOT_FROZEN",
            "arm": arm,
            "seed": seed,
            "epoch_index": epoch_index,
            "epoch_number": epoch_index + 1,
            "row_position": row_position,
            "score_loss_active": True,
            "cutoff_len": expected_cutoff,
            "padding_mode": "fixed_max_length",
            "truncated": False,
            "packing": False,
        }
        for field, value in expected.items():
            if row[field] != value:
                raise ValueError(
                    f"{arm}/seed {seed}/row {index}: {field} differs"
                )
        target_ids = list(row["target_token_ids"])
        if token_ids_sha256(target_ids) != row["target_token_ids_sha256"]:
            raise ValueError(f"{arm}/seed {seed}/row {index}: target hash differs")
        target_length = len(target_ids)
        sequence_length = int(row["sequence_token_count"])
        if sequence_length + int(row["padding_token_count"]) != expected_cutoff:
            raise ValueError(f"{arm}/seed {seed}/row {index}: padding differs")
        if sequence_length > expected_cutoff or target_length < 1:
            raise ValueError(f"{arm}/seed {seed}/row {index}: invalid length")

        score_positions = list(row["score_token_positions_in_target"])
        rationale_positions = list(row["rationale_token_positions_in_target"])
        if not score_positions:
            raise ValueError(f"{arm}/seed {seed}/row {index}: score mask empty")
        if set(score_positions) & set(rationale_positions):
            raise ValueError(f"{arm}/seed {seed}/row {index}: masks overlap")
        if any(position < 0 or position >= target_length for position in score_positions):
            raise ValueError(f"{arm}/seed {seed}/row {index}: score mask outside target")
        if any(
            position < 0 or position >= target_length
            for position in rationale_positions
        ):
            raise ValueError(
                f"{arm}/seed {seed}/row {index}: rationale mask outside target"
            )
        active = bool(row["rationale_active"])
        expected_boundary_padding = (
            config["active_rationale_unsupervised_boundary_padding"]
            if active
            else ""
        )
        if row["rationale_boundary_padding"] != expected_boundary_padding:
            raise ValueError(
                f"{arm}/seed {seed}/row {index}: rationale boundary differs"
            )
        if active != bool(rationale_positions):
            raise ValueError(
                f"{arm}/seed {seed}/row {index}: rationale mask/activity differ"
            )
        expected_weight = 1.0 if active else 0.0
        if float(row["rationale_block_weight"]) != expected_weight:
            raise ValueError(
                f"{arm}/seed {seed}/row {index}: rationale weight differs"
            )
        absolute_score_positions = list(row["score_token_positions"])
        absolute_rationale_positions = list(row["rationale_token_positions"])
        shift = absolute_score_positions[0] - score_positions[0]
        if absolute_score_positions != [
            shift + position for position in score_positions
        ]:
            raise ValueError(f"{arm}/seed {seed}/row {index}: score shift differs")
        if absolute_rationale_positions != [
            shift + position for position in rationale_positions
        ]:
            raise ValueError(
                f"{arm}/seed {seed}/row {index}: rationale shift differs"
            )
        if shift <= 0 or shift + target_length >= sequence_length:
            raise ValueError(
                f"{arm}/seed {seed}/row {index}: prompt/target/suffix boundary invalid"
            )


def _cross_arm_checks(
    manifests: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    base = manifests["S0"]
    invariant_fields = (
        "base_event_id",
        "seed",
        "epoch_index",
        "epoch_number",
        "row_position",
        "record_id",
        "prompt_cache_id",
        "prompt_token_ids_sha256",
        "score_target",
        "score_loss_active",
        "score_token_ids",
        "score_token_positions_in_target",
        "score_token_positions",
        "score_block_weight",
        "cutoff_len",
        "padding_mode",
        "packing",
        "truncated",
    )
    checks = {
        "same_row_count": all(len(rows) == len(base) for rows in manifests.values()),
        "same_base_event_order": True,
        "same_prompt_and_score_contract": True,
        "score_loss_active_for_every_event": True,
        "s0_rationale_always_inactive": True,
        "r2_r3_active_vector_identical": True,
    }
    for index, base_row in enumerate(base):
        for arm in ARMS:
            row = manifests[arm][index]
            if row["base_event_id"] != base_row["base_event_id"]:
                checks["same_base_event_order"] = False
            if any(row[field] != base_row[field] for field in invariant_fields):
                checks["same_prompt_and_score_contract"] = False
            if row["score_loss_active"] is not True:
                checks["score_loss_active_for_every_event"] = False
        if base_row["rationale_active"] or base_row["rationale_token_positions"]:
            checks["s0_rationale_always_inactive"] = False
        if (
            manifests["R2"][index]["rationale_active"]
            != manifests["R3"][index]["rationale_active"]
        ):
            checks["r2_r3_active_vector_identical"] = False
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"cross-arm checks failed: {failed}")
    return checks


def _frequency_counters(
    rows: list[dict[str, Any]],
) -> dict[str, Counter[str] | int]:
    active = [row for row in rows if row["rationale_active"]]
    return {
        "rationale_bytes": Counter(
            str(row["rationale_bytes_sha256"]) for row in active
        ),
        "rationale_token_ids": Counter(
            token_ids_sha256(list(row["rationale_token_ids"]))
            for row in active
        ),
        "target_bytes": Counter(str(row["target_bytes_sha256"]) for row in active),
        "target_token_ids": Counter(
            str(row["target_token_ids_sha256"]) for row in active
        ),
        "supervised_rationale_tokens": sum(
            len(row["rationale_token_positions"]) for row in active
        ),
    }


def _r2_r3_frequency_audit(
    r2: list[dict[str, Any]],
    r3: list[dict[str, Any]],
) -> dict[str, Any]:
    checkpoints = []
    for prefix_epochs in range(1, FORMAL_EPOCHS + 1):
        stop = prefix_epochs * EXPECTED_TRAIN_ROWS
        r2_counts = _frequency_counters(r2[:stop])
        r3_counts = _frequency_counters(r3[:stop])
        fields = (
            "rationale_bytes",
            "rationale_token_ids",
            "target_bytes",
            "target_token_ids",
        )
        l1 = {
            field: _counter_l1(r2_counts[field], r3_counts[field])
            for field in fields
        }
        checks = {
            **{f"{field}_counter_equal": value == 0 for field, value in l1.items()},
            "supervised_rationale_token_total_equal": (
                r2_counts["supervised_rationale_tokens"]
                == r3_counts["supervised_rationale_tokens"]
            ),
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise AssertionError(
                f"R2/R3 checkpoint prefix {prefix_epochs} failed: {failed}"
            )
        checkpoints.append(
            {
                "checkpoint_prefix_epochs": prefix_epochs,
                "row_events": stop,
                "active_rationale_events": sum(
                    bool(row["rationale_active"]) for row in r2[:stop]
                ),
                "checks": checks,
                "frequency_l1_difference": l1,
                "supervised_rationale_tokens": int(
                    r2_counts["supervised_rationale_tokens"]
                ),
                "counter_hashes": {
                    field: _counter_sha256(r2_counts[field])
                    for field in fields
                },
            }
        )

    individual_epochs = []
    for epoch_index in range(FORMAL_EPOCHS):
        start = epoch_index * EXPECTED_TRAIN_ROWS
        stop = start + EXPECTED_TRAIN_ROWS
        r2_counts = _frequency_counters(r2[start:stop])
        r3_counts = _frequency_counters(r3[start:stop])
        l1 = {
            field: _counter_l1(r2_counts[field], r3_counts[field])
            for field in (
                "rationale_bytes",
                "rationale_token_ids",
                "target_bytes",
                "target_token_ids",
            )
        }
        token_total_equal = (
            r2_counts["supervised_rationale_tokens"]
            == r3_counts["supervised_rationale_tokens"]
        )
        if any(l1.values()) or not token_total_equal:
            raise AssertionError(f"R2/R3 individual epoch {epoch_index + 1} differs")
        individual_epochs.append(
            {
                "epoch_number": epoch_index + 1,
                "row_events": EXPECTED_TRAIN_ROWS,
                "active_rationale_events": sum(
                    bool(row["rationale_active"]) for row in r2[start:stop]
                ),
                "frequency_l1_difference": l1,
                "supervised_rationale_tokens": int(
                    r2_counts["supervised_rationale_tokens"]
                ),
                "supervised_rationale_token_total_equal": token_total_equal,
            }
        )
    return {
        "individual_epochs": individual_epochs,
        "checkpoint_prefixes": checkpoints,
    }


def _training_budget(
    manifests: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    micro_batch = int(config["micro_batch_size_per_device"])
    accumulation = int(config["gradient_accumulation_steps"])
    cutoff = int(config["cutoff_len"])
    micro_batches_per_epoch = math.ceil(EXPECTED_TRAIN_ROWS / micro_batch)
    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / accumulation
    )
    per_arm = {}
    for arm, rows in manifests.items():
        per_arm[arm] = {
            "row_events": len(rows),
            "logical_epochs": FORMAL_EPOCHS,
            "fixed_padded_input_tokens": len(rows) * cutoff,
            "unpadded_sequence_tokens": sum(
                int(row["sequence_token_count"]) for row in rows
            ),
            "minimum_sequence_tokens": min(
                int(row["sequence_token_count"]) for row in rows
            ),
            "maximum_sequence_tokens": max(
                int(row["sequence_token_count"]) for row in rows
            ),
            "score_supervised_events": sum(
                bool(row["score_loss_active"]) for row in rows
            ),
            "score_supervised_tokens": sum(
                len(row["score_token_positions"]) for row in rows
            ),
            "rationale_supervised_events": sum(
                bool(row["rationale_active"]) for row in rows
            ),
            "rationale_supervised_tokens": sum(
                len(row["rationale_token_positions"]) for row in rows
            ),
        }
    invariant = {
        "row_events_equal": len({value["row_events"] for value in per_arm.values()}) == 1,
        "fixed_padded_input_tokens_equal": (
            len(
                {
                    value["fixed_padded_input_tokens"]
                    for value in per_arm.values()
                }
            )
            == 1
        ),
        "score_supervised_events_equal": (
            len(
                {
                    value["score_supervised_events"]
                    for value in per_arm.values()
                }
            )
            == 1
        ),
        "score_supervised_tokens_equal": (
            len(
                {
                    value["score_supervised_tokens"]
                    for value in per_arm.values()
                }
            )
            == 1
        ),
        "one_physical_manifest_pass": config["physical_manifest_passes"] == 1,
        "shuffle_disabled": config["shuffle"] is False,
        "packing_disabled": config["packing"] is False,
        "fixed_padding": config["padding"] == "fixed_max_length",
        "truncation_disabled": config["truncation"] is False,
        "logical_epoch_boundary_flush": (
            config["gradient_accumulation_boundary"]
            == "flush_and_reset_at_each_logical_epoch"
        ),
    }
    if not all(invariant.values()):
        failed = [name for name, passed in invariant.items() if not passed]
        raise AssertionError(f"training-budget checks failed: {failed}")
    return {
        "checks": invariant,
        "cutoff_len": cutoff,
        "micro_batch_size_per_device": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "micro_batches_per_logical_epoch": micro_batches_per_epoch,
        "optimizer_steps_per_logical_epoch": optimizer_steps_per_epoch,
        "optimizer_steps_total": optimizer_steps_per_epoch * FORMAL_EPOCHS,
        "last_accumulation_group_micro_batches_per_epoch": (
            micro_batches_per_epoch % accumulation or accumulation
        ),
        "per_arm": per_arm,
        "note": (
            "Different rationale-supervised totals across S0/R1/R2-R3 are the "
            "intended intervention. Equal fixed padded tokens, event order, "
            "score supervision, and optimizer steps control training budget."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reject_eval_path(args.config)
    if not args.config.exists():
        raise FileNotFoundError(args.config)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "CANDIDATE_NOT_FROZEN":
        raise ValueError("manifest config must remain candidate-only")
    if (
        config.get("manifest_freeze_allowed")
        or config.get("smoke_training_allowed")
        or config.get("formal_training_allowed")
    ):
        raise PermissionError("candidate config must not authorize training")
    if config.get("dev_accessed") or config.get("test_accessed"):
        raise PermissionError("candidate config indicates evaluation access")
    for path in (*SOURCE_PATHS.values(), *UPSTREAM_LOCK_PATHS.values()):
        reject_eval_path(path)
        if not path.exists():
            raise FileNotFoundError(path)

    prompt_cache_path = args.output_dir / "data/shared_prompt_cache.jsonl"
    reject_eval_path(prompt_cache_path)
    if not prompt_cache_path.exists():
        raise FileNotFoundError(prompt_cache_path)
    prompt_cache = read_jsonl(prompt_cache_path, protect_split=True)
    if len(prompt_cache) != EXPECTED_TRAIN_ROWS:
        raise ValueError("prompt cache does not contain exactly the train rows")
    if [int(row["row_position"]) for row in prompt_cache] != list(
        range(EXPECTED_TRAIN_ROWS)
    ):
        raise ValueError("prompt cache row order differs")

    seed_reports = []
    private_hashes: dict[str, Any] = {
        "shared_prompt_cache": file_sha256(prompt_cache_path),
        "manifests_by_seed": {},
    }
    for seed in FORMAL_SEEDS:
        manifests: dict[str, list[dict[str, Any]]] = {}
        private_hashes["manifests_by_seed"][f"seed{seed}"] = {}
        for arm in ARMS:
            path = (
                args.output_dir
                / "data"
                / f"training_manifest_{arm.lower()}_seed{seed}.jsonl"
            )
            reject_eval_path(path)
            if not path.exists():
                raise FileNotFoundError(path)
            rows = read_jsonl(path, protect_split=True)
            _require_exact_manifest(rows, arm=arm, seed=seed, config=config)
            manifests[arm] = rows
            private_hashes["manifests_by_seed"][f"seed{seed}"][arm] = (
                file_sha256(path)
            )

        cross_arm = _cross_arm_checks(manifests)
        frequency = _r2_r3_frequency_audit(manifests["R2"], manifests["R3"])
        budget = _training_budget(manifests, config)
        seed_reports.append(
            {
                "seed": seed,
                "checks": cross_arm,
                "row_events_per_arm": EXPECTED_EVENTS_PER_SEED,
                "ordered_base_event_vector_sha256": _vector_sha256(
                    row["base_event_id"] for row in manifests["S0"]
                ),
                "score_target_vector_sha256": _vector_sha256(
                    row["score_target"] for row in manifests["S0"]
                ),
                "r2_r3_frequency_control": frequency,
                "training_budget": budget,
            }
        )

    report = {
        "status": "MATERIALIZED_MANIFEST_CANDIDATE_AUDITED_NOT_FROZEN",
        "scope": (
            "train-only candidate S0/R1/R2/R3 target bytes, token IDs, "
            "loss masks, event order, checkpoint-prefix exposure, and budget"
        ),
        "formal_seeds": list(FORMAL_SEEDS),
        "arms": list(ARMS),
        "logical_epochs": FORMAL_EPOCHS,
        "rows_per_logical_epoch": EXPECTED_TRAIN_ROWS,
        "seed_reports": seed_reports,
        "private_artifact_hashes": private_hashes,
        "config_sha256": file_sha256(args.config),
        "source_hashes": {
            name: file_sha256(path) for name, path in SOURCE_PATHS.items()
        },
        "upstream_lock_hashes": {
            name: file_sha256(path)
            for name, path in UPSTREAM_LOCK_PATHS.items()
        },
        "manifest_freeze_allowed": False,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    report_path = (
        args.output_dir / "audit/materialized_manifest_candidate_report.json"
    )
    lock_path = (
        args.output_dir / "protocol/materialized_manifest_candidate_lock.json"
    )
    write_json(report_path, report)
    write_json(
        lock_path,
        {
            "status": report["status"],
            "formal_seeds": report["formal_seeds"],
            "arms": report["arms"],
            "logical_epochs": report["logical_epochs"],
            "rows_per_logical_epoch": report["rows_per_logical_epoch"],
            "private_artifact_hashes": private_hashes,
            "candidate_report_sha256": file_sha256(report_path),
            "config_sha256": report["config_sha256"],
            "source_hashes": report["source_hashes"],
            "upstream_lock_hashes": report["upstream_lock_hashes"],
            "manifest_freeze_allowed": False,
            "smoke_training_allowed": False,
            "formal_training_allowed": False,
            "dev_accessed": False,
            "test_accessed": False,
            "training_used": False,
        },
    )


if __name__ == "__main__":
    main()
