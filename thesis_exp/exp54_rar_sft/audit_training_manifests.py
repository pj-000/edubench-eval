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
    DEFAULT_ALL_REFERENCES,
    DEFAULT_CONSISTENT_REFERENCES,
    DEFAULT_OUTPUT,
    DEFAULT_TRAIN,
    EXPECTED_EVENTS_PER_SEED,
)
from thesis_exp.exp54_rar_sft.block_loss import FixedRARCollator
from thesis_exp.exp54_rar_sft.manifest_source_contract import (
    ARMS,
    index_unique_rows,
    reference_indexes,
    resolve_expected_arm_source,
    source_fingerprint,
)
from thesis_exp.exp54_rar_sft.reference_schedule import (
    EXPECTED_TRAIN_ROWS,
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
    validate_schedule_events,
)
from thesis_exp.exp54_rar_sft.training_contract import (
    CONTRACT_VERSION,
    DEFAULT_RUBRIC_REGISTRY,
    RATIONALE_BOUNDARY_PADDING,
    build_prompt_cache_row,
    load_locked_tokenizer,
    materialize_sequence,
    require_canonical_positions,
    sha256_bytes,
    tokenize_target,
    token_ids_sha256,
    tokenizer_boundary_probes,
)


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
    "prompt_cleaning_source": (
        REPO_ROOT / "thesis_exp/src/edujudge/exp02/build_exp02_dataset.py"
    ),
    "shared_reference_schedule": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/reference_schedule.py"
    ),
    "canonical_rubric_registry": DEFAULT_RUBRIC_REGISTRY,
    "manifest_source_contract": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/manifest_source_contract.py"
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


def _require_file_hash(path: Path, expected_hash: str, *, name: str) -> str:
    reject_eval_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{name} differs from its upstream lock")
    return actual_hash


def _load_seed_source_context(
    *,
    output_dir: Path,
    seed: int,
    consistent_reference_rows: list[dict[str, Any]],
    schedule_lock: dict[str, Any],
    donor_lock: dict[str, Any],
    mask_lock: dict[str, Any],
) -> dict[str, Any]:
    schedule_path = output_dir / "data" / f"base_event_schedule_seed{seed}.jsonl"
    donor_path = output_dir / "data" / f"r2_event_donor_map_seed{seed}.jsonl"
    r2_mask_path = output_dir / "data" / f"r2_event_active_mask_seed{seed}.jsonl"
    r3_mask_path = output_dir / "data" / f"r3_event_active_mask_seed{seed}.jsonl"
    schedule_hash = _require_file_hash(
        schedule_path,
        str(
            schedule_lock["schedule_file_hashes"][
                f"base_event_schedule_seed{seed}"
            ]
        ),
        name=f"seed {seed} base schedule",
    )
    donor_hash = _require_file_hash(
        donor_path,
        str(donor_lock["event_donor_map_sha256_by_seed"][f"seed{seed}"]),
        name=f"seed {seed} event donor map",
    )
    expected_masks = mask_lock["event_mask_sha256_by_seed"][f"seed{seed}"]
    r2_mask_hash = _require_file_hash(
        r2_mask_path,
        str(expected_masks["r2"]),
        name=f"seed {seed} R2 event mask",
    )
    r3_mask_hash = _require_file_hash(
        r3_mask_path,
        str(expected_masks["r3"]),
        name=f"seed {seed} R3 event mask",
    )
    if r2_mask_path.read_bytes() != r3_mask_path.read_bytes():
        raise ValueError(f"seed {seed}: R2/R3 event masks are not byte-identical")

    base_events = read_jsonl(schedule_path, protect_split=True)
    donor_rows = read_jsonl(donor_path, protect_split=True)
    r2_mask_rows = read_jsonl(r2_mask_path, protect_split=True)
    r3_mask_rows = read_jsonl(r3_mask_path, protect_split=True)
    validate_schedule_events(
        consistent_reference_rows,
        base_events,
        seed=seed,
        epochs=FORMAL_EPOCHS,
    )
    if r2_mask_rows != r3_mask_rows:
        raise ValueError(f"seed {seed}: parsed R2/R3 masks differ")
    if (
        len(base_events) != EXPECTED_EVENTS_PER_SEED
        or len(donor_rows) != EXPECTED_EVENTS_PER_SEED
        or len(r2_mask_rows) != EXPECTED_EVENTS_PER_SEED
    ):
        raise ValueError(f"seed {seed}: source artifact row count differs")
    donor_by_event = index_unique_rows(
        donor_rows,
        "recipient_event_id",
        source=f"seed {seed} donor map",
    )
    mask_by_event = index_unique_rows(
        r2_mask_rows,
        "base_event_id",
        source=f"seed {seed} event mask",
    )
    base_ids = [str(event["event_id"]) for event in base_events]
    if set(donor_by_event) != set(base_ids) or set(mask_by_event) != set(base_ids):
        raise ValueError(f"seed {seed}: source artifacts do not cover base events")
    for event, mask in zip(base_events, r2_mask_rows, strict=True):
        for event_field, mask_field in (
            ("event_id", "base_event_id"),
            ("seed", "seed"),
            ("epoch_index", "epoch_index"),
            ("epoch_number", "epoch_number"),
            ("row_position", "row_position"),
            ("record_id", "record_id"),
            ("selected_reference_id", "base_selected_reference_id"),
        ):
            if event[event_field] != mask[mask_field]:
                raise ValueError(
                    f"seed {seed}/{event['event_id']}: mask provenance differs"
                )
    return {
        "base_events": base_events,
        "donor_by_event": donor_by_event,
        "mask_by_event": mask_by_event,
        "artifact_hashes": {
            "base_schedule_sha256": schedule_hash,
            "event_donor_map_sha256": donor_hash,
            "r2_event_mask_sha256": r2_mask_hash,
            "r3_event_mask_sha256": r3_mask_hash,
        },
    }


def _validate_prompt_cache(
    tokenizer: Any,
    train_rows: list[dict[str, Any]],
    prompt_cache: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if len(train_rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError("train split does not contain exactly 2,654 rows")
    if len(prompt_cache) != EXPECTED_TRAIN_ROWS:
        raise ValueError("prompt cache does not contain exactly the train rows")
    prompt_by_id: dict[str, dict[str, Any]] = {}
    metric_ids: set[str] = set()
    metric_language_pairs: set[tuple[str, str]] = set()
    for row_position, (train_row, prompt_row) in enumerate(
        zip(train_rows, prompt_cache, strict=True)
    ):
        expected = build_prompt_cache_row(
            tokenizer,
            train_row,
            row_position=row_position,
        )
        if prompt_row != expected:
            raise ValueError(
                f"prompt cache row {row_position} differs from independent rebuild"
            )
        prompt_id = str(prompt_row["prompt_cache_id"])
        if prompt_id in prompt_by_id:
            raise ValueError(f"duplicate prompt_cache_id: {prompt_id}")
        prompt_by_id[prompt_id] = prompt_row
        metric_id = str(train_row["metric_id"])
        language = str(train_row["language"])
        metric_ids.add(metric_id)
        metric_language_pairs.add((metric_id, language))
    return prompt_by_id, {
        "train_rows_validated": len(train_rows),
        "metric_count": len(metric_ids),
        "metric_language_entry_count": len(metric_language_pairs),
        "expected_metric_count": 12,
        "expected_metric_language_entry_count": 24,
        "all_rows_match_canonical_registry": True,
    }


def _compare_derived_fields(
    actual: dict[str, Any],
    expected: dict[str, Any],
    fields: tuple[str, ...],
    *,
    context: str,
) -> None:
    for field in fields:
        if actual.get(field) != expected[field]:
            raise ValueError(f"{context}: derived field differs: {field}")


def verify_manifest_row(
    tokenizer: Any,
    row: dict[str, Any],
    prompt_by_id: dict[str, dict[str, Any]],
    expected_train_row: dict[str, Any],
    expected_base_event: dict[str, Any],
    expected_source: dict[str, Any],
    *,
    arm: str,
    seed: int,
    index: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    epoch_index, row_position = divmod(index, EXPECTED_TRAIN_ROWS)
    expected_metadata = {
        "contract_version": CONTRACT_VERSION,
        "candidate_status": "CANDIDATE_NOT_FROZEN",
        "arm": arm,
        "base_event_id": str(expected_base_event["event_id"]),
        "seed": seed,
        "epoch_index": epoch_index,
        "epoch_number": epoch_index + 1,
        "row_position": row_position,
        "record_id": str(expected_train_row["record_id"]),
        "score_loss_active": True,
        "base_selected_reference_id": expected_base_event[
            "selected_reference_id"
        ],
        "cutoff_len": int(config["cutoff_len"]),
        "padding_mode": "fixed_max_length",
        "truncated": False,
        "packing": False,
    }
    context = f"{arm}/seed {seed}/row {index}"
    for field, value in expected_metadata.items():
        if row.get(field) != value:
            raise ValueError(f"{context}: metadata differs: {field}")
    if row.get("score_loss_active") is not True:
        raise ValueError(f"{context}: score loss must be the boolean true")
    event_metadata = {
        "seed": seed,
        "epoch_index": epoch_index,
        "epoch_number": epoch_index + 1,
        "row_position": row_position,
        "record_id": str(expected_train_row["record_id"]),
        "label_5": int(expected_train_row["label_5"]),
        "metric_id": str(expected_train_row["metric_id"]),
        "language": str(expected_train_row["language"]),
    }
    for field, value in event_metadata.items():
        if expected_base_event[field] != value:
            raise ValueError(f"{context}: base schedule differs from locked train")
    source_fields = (
        "rationale",
        "rationale_active",
        "arm_selected_reference_id",
        "arm_rationale_source_event_id",
        "inactive_reason",
    )
    for field in source_fields:
        if row.get(field) != expected_source[field]:
            raise ValueError(f"{context}: arm source differs: {field}")

    prompt_id = str(row.get("prompt_cache_id") or "")
    prompt = prompt_by_id.get(prompt_id)
    if prompt is None:
        raise ValueError(f"{context}: prompt cache ID is missing")
    if str(row.get("record_id")) != str(prompt["record_id"]):
        raise ValueError(f"{context}: prompt belongs to another record")
    if int(row["row_position"]) != int(prompt["row_position"]):
        raise ValueError(f"{context}: prompt row position differs")
    if (
        row.get("prompt_token_ids_sha256")
        != prompt["prompt_token_ids_sha256"]
    ):
        raise ValueError(f"{context}: prompt token hash differs")

    if not isinstance(row.get("rationale_active"), bool):
        raise ValueError(f"{context}: rationale_active must be boolean")
    active = row["rationale_active"]
    rationale = row.get("rationale")
    if not isinstance(rationale, str):
        raise ValueError(f"{context}: rationale source must be a string")
    expected_boundary = RATIONALE_BOUNDARY_PADDING if active else ""
    if row.get("rationale_boundary_padding") != expected_boundary:
        raise ValueError(f"{context}: rationale boundary padding differs")
    if expected_boundary != (
        config["active_rationale_unsupervised_boundary_padding"] if active else ""
    ):
        raise ValueError(f"{context}: config boundary padding differs")
    if (
        not isinstance(row.get("score_target"), int)
        or isinstance(row.get("score_target"), bool)
    ):
        raise ValueError(f"{context}: score target must be an integer")
    score = row["score_target"]
    expected_score = int(expected_train_row["label_5"])
    if score != expected_score:
        raise ValueError(f"{context}: score target differs from locked label_5")
    expected_target = tokenize_target(
        tokenizer,
        score=score,
        rationale=rationale,
        rationale_active=active,
    )
    try:
        parsed_target = json.loads(str(row.get("target_text")))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context}: target text is not valid JSON") from exc
    if list(parsed_target) != ["score", "rationale"]:
        raise ValueError(f"{context}: target JSON fields/order differ")
    if parsed_target["score"] != score:
        raise ValueError(f"{context}: parsed target score differs")
    expected_visible_rationale = rationale + expected_boundary if active else ""
    if parsed_target["rationale"] != expected_visible_rationale:
        raise ValueError(f"{context}: parsed target rationale differs")
    if (
        sha256_bytes(str(row["target_text"]).encode("utf-8"))
        != row.get("target_bytes_sha256")
    ):
        raise ValueError(f"{context}: target bytes hash differs")
    if (
        sha256_bytes(rationale.encode("utf-8"))
        != row.get("rationale_bytes_sha256")
    ):
        raise ValueError(f"{context}: rationale bytes hash differs")

    for field in (
        "score_token_positions_in_target",
        "rationale_token_positions_in_target",
    ):
        require_canonical_positions(row.get(field), field=f"{context}: {field}")
    _compare_derived_fields(
        row,
        expected_target,
        (
            "target_text",
            "target_bytes_sha256",
            "target_token_ids",
            "target_token_ids_sha256",
            "score_token_positions_in_target",
            "rationale_token_positions_in_target",
            "score_token_ids",
            "rationale_token_ids",
        ),
        context=context,
    )

    expected_sequence = materialize_sequence(
        tokenizer,
        prompt,
        expected_target,
    )
    for field in ("score_token_positions", "rationale_token_positions"):
        require_canonical_positions(row.get(field), field=f"{context}: {field}")
    _compare_derived_fields(
        row,
        expected_sequence,
        (
            "sequence_token_count",
            "full_token_ids_sha256",
            "assistant_suffix_token_ids",
            "assistant_suffix_token_ids_sha256",
            "score_token_positions",
            "rationale_token_positions",
            "score_mask_sha256",
            "rationale_mask_sha256",
        ),
        context=context,
    )
    full_ids = (
        list(prompt["prompt_token_ids"])
        + list(expected_target["target_token_ids"])
        + list(expected_sequence["assistant_suffix_token_ids"])
    )
    if token_ids_sha256(full_ids) != row["full_token_ids_sha256"]:
        raise ValueError(f"{context}: independently rebuilt sequence hash differs")
    if len(full_ids) != int(row["sequence_token_count"]):
        raise ValueError(f"{context}: independently rebuilt sequence length differs")

    cutoff = int(config["cutoff_len"])
    padding_count = cutoff - len(full_ids)
    if padding_count < 0:
        raise ValueError(f"{context}: sequence exceeds cutoff")
    if int(row["padding_token_count"]) != padding_count:
        raise ValueError(f"{context}: padding count differs")
    score_positions = set(expected_sequence["score_token_positions"])
    rationale_positions = set(expected_sequence["rationale_token_positions"])
    if score_positions & rationale_positions:
        raise ValueError(f"{context}: rebuilt masks overlap")
    target_start = len(prompt["prompt_token_ids"])
    target_end = target_start + len(expected_target["target_token_ids"])
    supervised = score_positions | rationale_positions
    if any(position < target_start or position >= target_end for position in supervised):
        raise ValueError(f"{context}: non-target token receives supervision")
    if any(position >= len(full_ids) for position in supervised):
        raise ValueError(f"{context}: suffix/padding receives supervision")
    if not score_positions:
        raise ValueError(f"{context}: score mask is empty")
    if bool(rationale_positions) != active:
        raise ValueError(f"{context}: rationale mask activity differs")
    expected_weight = 1.0 if active else 0.0
    if float(row["rationale_block_weight"]) != expected_weight:
        raise ValueError(f"{context}: rationale block weight differs")
    if float(row["score_block_weight"]) != 1.0:
        raise ValueError(f"{context}: score block weight differs")

    verified = dict(row)
    verified["_audit_sequence_token_count"] = len(full_ids)
    verified["_audit_padding_token_count"] = padding_count
    verified["_audit_score_mask_token_count"] = len(score_positions)
    verified["_audit_rationale_mask_token_count"] = len(rationale_positions)
    return verified


def _require_exact_manifest(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    prompt_by_id: dict[str, dict[str, Any]],
    train_rows: list[dict[str, Any]],
    base_events: list[dict[str, Any]],
    all_references_by_record: dict[str, dict[str, Any]],
    consistent_references_by_id: dict[str, dict[str, Any]],
    donor_by_event: dict[str, dict[str, Any]],
    mask_by_event: dict[str, dict[str, Any]],
    *,
    arm: str,
    seed: int,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    if len(rows) != EXPECTED_EVENTS_PER_SEED:
        raise ValueError(f"{arm}/seed {seed}: wrong manifest row count")
    if len({str(row["base_event_id"]) for row in rows}) != len(rows):
        raise ValueError(f"{arm}/seed {seed}: duplicate base event ID")
    if len(base_events) != len(rows):
        raise ValueError(f"{arm}/seed {seed}: base schedule length differs")
    verified_rows: list[dict[str, Any]] = []
    source_vector: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        expected_train_row = train_rows[index % EXPECTED_TRAIN_ROWS]
        expected_base_event = base_events[index]
        expected_source = resolve_expected_arm_source(
            arm=arm,
            train_row=expected_train_row,
            base_event=expected_base_event,
            all_references_by_record=all_references_by_record,
            consistent_references_by_id=consistent_references_by_id,
            donor_by_event=donor_by_event,
            mask_by_event=mask_by_event,
        )
        verified_rows.append(
            verify_manifest_row(
            tokenizer,
            row,
            prompt_by_id,
            expected_train_row,
            expected_base_event,
            expected_source,
            arm=arm,
            seed=seed,
            index=index,
            config=config,
        )
        )
        source_vector.append(source_fingerprint(expected_source))
    return verified_rows, _vector_sha256(source_vector)


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
    active_rows = [row for row in rows if row["rationale_active"]]
    return {
        "active_rationale_bytes": Counter(
            str(row["rationale_bytes_sha256"]) for row in active_rows
        ),
        "active_rationale_token_ids": Counter(
            token_ids_sha256(list(row["rationale_token_ids"]))
            for row in active_rows
        ),
        "all_event_target_bytes": Counter(
            str(row["target_bytes_sha256"]) for row in rows
        ),
        "all_event_target_token_ids": Counter(
            str(row["target_token_ids_sha256"]) for row in rows
        ),
        "supervised_rationale_tokens": sum(
            int(row["_audit_rationale_mask_token_count"])
            for row in active_rows
        ),
        "rationale_active_vector_sha256": _vector_sha256(
            bool(row["rationale_active"]) for row in rows
        ),
        "unpadded_sequence_tokens": sum(
            int(row["_audit_sequence_token_count"]) for row in rows
        ),
        "fixed_padded_tokens": sum(
            int(row["_audit_sequence_token_count"])
            + int(row["_audit_padding_token_count"])
            for row in rows
        ),
    }


def _period_control_report(
    r2_rows: list[dict[str, Any]],
    r3_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    r2_counts = _frequency_counters(r2_rows)
    r3_counts = _frequency_counters(r3_rows)
    counter_fields = (
        "active_rationale_bytes",
        "active_rationale_token_ids",
        "all_event_target_bytes",
        "all_event_target_token_ids",
    )
    l1 = {
        field: _counter_l1(r2_counts[field], r3_counts[field])
        for field in counter_fields
    }
    checks = {
        **{f"{field}_counter_equal": value == 0 for field, value in l1.items()},
        "rationale_active_vector_equal": (
            r2_counts["rationale_active_vector_sha256"]
            == r3_counts["rationale_active_vector_sha256"]
        ),
        "supervised_rationale_token_total_equal": (
            r2_counts["supervised_rationale_tokens"]
            == r3_counts["supervised_rationale_tokens"]
        ),
        "unpadded_sequence_token_total_equal": (
            r2_counts["unpadded_sequence_tokens"]
            == r3_counts["unpadded_sequence_tokens"]
        ),
        "fixed_padded_token_total_equal": (
            r2_counts["fixed_padded_tokens"]
            == r3_counts["fixed_padded_tokens"]
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"R2/R3 period control failed: {failed}")
    return {
        "checks": checks,
        "frequency_l1_difference": l1,
        "counter_hashes": {
            field: _counter_sha256(r2_counts[field])
            for field in counter_fields
        },
        "rationale_active_vector_sha256_R2": r2_counts[
            "rationale_active_vector_sha256"
        ],
        "rationale_active_vector_sha256_R3": r3_counts[
            "rationale_active_vector_sha256"
        ],
        "supervised_rationale_token_total_R2": int(
            r2_counts["supervised_rationale_tokens"]
        ),
        "supervised_rationale_token_total_R3": int(
            r3_counts["supervised_rationale_tokens"]
        ),
        "unpadded_sequence_token_total_R2": int(
            r2_counts["unpadded_sequence_tokens"]
        ),
        "unpadded_sequence_token_total_R3": int(
            r3_counts["unpadded_sequence_tokens"]
        ),
        "fixed_padded_token_total_R2": int(r2_counts["fixed_padded_tokens"]),
        "fixed_padded_token_total_R3": int(r3_counts["fixed_padded_tokens"]),
    }


def _r2_r3_frequency_audit(
    r2: list[dict[str, Any]],
    r3: list[dict[str, Any]],
) -> dict[str, Any]:
    checkpoints = []
    for prefix_epochs in range(1, FORMAL_EPOCHS + 1):
        stop = prefix_epochs * EXPECTED_TRAIN_ROWS
        control = _period_control_report(r2[:stop], r3[:stop])
        checkpoints.append(
            {
                "checkpoint_prefix_epochs": prefix_epochs,
                "row_events": stop,
                "active_rationale_events": sum(
                    bool(row["rationale_active"]) for row in r2[:stop]
                ),
                **control,
            }
        )

    individual_epochs = []
    for epoch_index in range(FORMAL_EPOCHS):
        start = epoch_index * EXPECTED_TRAIN_ROWS
        stop = start + EXPECTED_TRAIN_ROWS
        control = _period_control_report(r2[start:stop], r3[start:stop])
        individual_epochs.append(
            {
                "epoch_number": epoch_index + 1,
                "row_events": EXPECTED_TRAIN_ROWS,
                "active_rationale_events": sum(
                    bool(row["rationale_active"]) for row in r2[start:stop]
                ),
                **control,
            }
        )
    return {
        "individual_epochs": individual_epochs,
        "checkpoint_prefixes": checkpoints,
    }


def _training_budget(
    manifests: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    *,
    pad_token_id: int,
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
        collator = FixedRARCollator(
            pad_token_id=pad_token_id,
            cutoff_len=cutoff,
        )
        collator_score_tokens = 0
        collator_rationale_tokens = 0
        collator_padded_tokens = 0
        for start in range(0, len(rows), 128):
            batch_rows = rows[start : start + 128]
            batch = collator(
                [
                    {
                        "input_ids": [0]
                        * int(row["_audit_sequence_token_count"]),
                        "score_token_positions": row[
                            "score_token_positions"
                        ],
                        "rationale_token_positions": row[
                            "rationale_token_positions"
                        ],
                        "rationale_active": row["rationale_active"],
                        "cutoff_len": cutoff,
                    }
                    for row in batch_rows
                ]
            )
            collator_score_tokens += int(batch["score_mask"].sum().item())
            collator_rationale_tokens += int(
                batch["rationale_mask"].sum().item()
            )
            collator_padded_tokens += int(batch["input_ids"].numel())
        audited_score_tokens = sum(
            int(row["_audit_score_mask_token_count"]) for row in rows
        )
        audited_rationale_tokens = sum(
            int(row["_audit_rationale_mask_token_count"]) for row in rows
        )
        expected_padded_tokens = len(rows) * cutoff
        if (
            collator_score_tokens != audited_score_tokens
            or collator_rationale_tokens != audited_rationale_tokens
            or collator_padded_tokens != expected_padded_tokens
        ):
            raise AssertionError(f"{arm}: collator and rebuilt mask totals differ")
        per_arm[arm] = {
            "row_events": len(rows),
            "logical_epochs": FORMAL_EPOCHS,
            "fixed_padded_input_tokens": expected_padded_tokens,
            "unpadded_sequence_tokens": sum(
                int(row["_audit_sequence_token_count"]) for row in rows
            ),
            "minimum_sequence_tokens": min(
                int(row["_audit_sequence_token_count"]) for row in rows
            ),
            "maximum_sequence_tokens": max(
                int(row["_audit_sequence_token_count"]) for row in rows
            ),
            "score_supervised_events": sum(
                bool(row["score_loss_active"]) for row in rows
            ),
            "score_supervised_tokens": audited_score_tokens,
            "rationale_supervised_events": sum(
                bool(row["rationale_active"]) for row in rows
            ),
            "rationale_supervised_tokens": audited_rationale_tokens,
            "collator_boolean_score_mask_tokens": collator_score_tokens,
            "collator_boolean_rationale_mask_tokens": (
                collator_rationale_tokens
            ),
            "collator_fixed_padded_tokens": collator_padded_tokens,
            "collator_totals_match_independent_rebuild": True,
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
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument(
        "--all-references",
        type=Path,
        default=DEFAULT_ALL_REFERENCES,
    )
    parser.add_argument(
        "--consistent-references",
        type=Path,
        default=DEFAULT_CONSISTENT_REFERENCES,
    )
    parser.add_argument(
        "--tokenizer-report",
        type=Path,
        default=UPSTREAM_LOCK_PATHS["tokenizer_report_seed42"],
    )
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.config,
        args.train,
        args.all_references,
        args.consistent_references,
        args.tokenizer_report,
    ):
        reject_eval_path(path)
        if not path.exists():
            raise FileNotFoundError(path)
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
    reference_lock = json.loads(
        UPSTREAM_LOCK_PATHS["reference_set_lock"].read_text(encoding="utf-8")
    )
    if file_sha256(args.train) != reference_lock["input"][
        "rar0_source_hashes"
    ]["train"]:
        raise ValueError("train split differs from the frozen reference lock")
    all_reference_hash = _require_file_hash(
        args.all_references,
        str(reference_lock["output_hashes"]["all_rater_reference_sets"]),
        name="all-rater reference inventory",
    )
    consistent_reference_hash = _require_file_hash(
        args.consistent_references,
        str(
            reference_lock["output_hashes"][
                "label_consistent_reference_sets"
            ]
        ),
        name="label-consistent reference inventory",
    )
    schedule_lock = json.loads(
        UPSTREAM_LOCK_PATHS["base_schedule_lock"].read_text(encoding="utf-8")
    )
    donor_lock = json.loads(
        UPSTREAM_LOCK_PATHS["event_donor_lock"].read_text(encoding="utf-8")
    )
    mask_lock = json.loads(
        UPSTREAM_LOCK_PATHS["event_mask_lock"].read_text(encoding="utf-8")
    )
    tokenizer_report = json.loads(
        args.tokenizer_report.read_text(encoding="utf-8")
    )
    tokenizer = load_locked_tokenizer(args.tokenizer_path, tokenizer_report)
    train_rows = read_jsonl(args.train, protect_split=True)
    all_reference_rows = read_jsonl(args.all_references, protect_split=True)
    consistent_reference_rows = read_jsonl(
        args.consistent_references,
        protect_split=True,
    )
    all_references_by_record, _all_references_by_id = reference_indexes(
        all_reference_rows,
        source="all-rater references",
    )
    (
        consistent_references_by_record,
        consistent_references_by_id,
    ) = reference_indexes(
        consistent_reference_rows,
        source="label-consistent references",
    )
    train_ids = [str(row["record_id"]) for row in train_rows]
    if (
        train_ids != list(all_references_by_record)
        or train_ids != list(consistent_references_by_record)
    ):
        raise ValueError("train and reference inventories differ in row order")

    prompt_cache_path = args.output_dir / "data/shared_prompt_cache.jsonl"
    reject_eval_path(prompt_cache_path)
    if not prompt_cache_path.exists():
        raise FileNotFoundError(prompt_cache_path)
    prompt_cache = read_jsonl(prompt_cache_path, protect_split=True)
    prompt_by_id, rubric_validation = _validate_prompt_cache(
        tokenizer,
        train_rows,
        prompt_cache,
    )
    if (
        rubric_validation["metric_count"] != 12
        or rubric_validation["metric_language_entry_count"] != 24
    ):
        raise ValueError("canonical rubric registry coverage differs")
    boundary_probes = tokenizer_boundary_probes(tokenizer)

    seed_reports = []
    private_hashes: dict[str, Any] = {
        "shared_prompt_cache": file_sha256(prompt_cache_path),
        "all_rater_reference_inventory": all_reference_hash,
        "label_consistent_reference_inventory": consistent_reference_hash,
        "manifests_by_seed": {},
        "source_artifacts_by_seed": {},
    }
    locked_score_vector = [
        int(row["label_5"])
        for _epoch_index in range(FORMAL_EPOCHS)
        for row in train_rows
    ]
    locked_score_vector_sha256 = _vector_sha256(locked_score_vector)
    for seed in FORMAL_SEEDS:
        source_context = _load_seed_source_context(
            output_dir=args.output_dir,
            seed=seed,
            consistent_reference_rows=consistent_reference_rows,
            schedule_lock=schedule_lock,
            donor_lock=donor_lock,
            mask_lock=mask_lock,
        )
        private_hashes["source_artifacts_by_seed"][f"seed{seed}"] = (
            source_context["artifact_hashes"]
        )
        manifests: dict[str, list[dict[str, Any]]] = {}
        expected_source_hashes: dict[str, str] = {}
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
            manifests[arm], expected_source_hashes[arm] = (
                _require_exact_manifest(
                rows,
                tokenizer,
                prompt_by_id,
                train_rows,
                source_context["base_events"],
                all_references_by_record,
                consistent_references_by_id,
                source_context["donor_by_event"],
                source_context["mask_by_event"],
                arm=arm,
                seed=seed,
                config=config,
            )
            )
            private_hashes["manifests_by_seed"][f"seed{seed}"][arm] = (
                file_sha256(path)
            )

        cross_arm = _cross_arm_checks(manifests)
        frequency = _r2_r3_frequency_audit(manifests["R2"], manifests["R3"])
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("locked tokenizer lacks pad/eos token ID")
        budget = _training_budget(
            manifests,
            config,
            pad_token_id=int(pad_token_id),
        )
        arm_score_hashes = {
            arm: _vector_sha256(row["score_target"] for row in rows)
            for arm, rows in manifests.items()
        }
        score_targets_equal_locked_train = all(
            value == locked_score_vector_sha256
            for value in arm_score_hashes.values()
        )
        if not score_targets_equal_locked_train:
            raise AssertionError(f"seed {seed}: score targets differ from train")
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
                "locked_train_score_vector_sha256": (
                    locked_score_vector_sha256
                ),
                "score_target_vector_sha256_by_arm": arm_score_hashes,
                "score_targets_equal_locked_train": (
                    score_targets_equal_locked_train
                ),
                "semantic_source_audit": {
                    "base_schedule_rows_verified": EXPECTED_EVENTS_PER_SEED,
                    "arm_source_rows_verified": {
                        arm: EXPECTED_EVENTS_PER_SEED for arm in ARMS
                    },
                    "r1_schedule_source_mismatches": 0,
                    "r2_donor_backlink_mismatches": 0,
                    "r3_aligned_reference_mismatches": 0,
                    "r2_r3_mask_mismatches": 0,
                    "expected_source_vector_sha256_by_arm": (
                        expected_source_hashes
                    ),
                },
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
        "independent_reconstruction": {
            "prompt_cache_rebuilt_from_locked_train_and_tokenizer": True,
            "target_and_masks_rebuilt_from_raw_manifest_score_rationale": True,
            "chat_sequence_rebuilt_from_prompt_target_and_suffix": True,
            "padded_boolean_mask_totals_recomputed": True,
            "score_targets_bound_to_locked_train_labels": True,
            "arm_sources_bound_to_locked_reference_schedule_donor_mask": True,
        },
        "locked_train_score_vector_sha256": locked_score_vector_sha256,
        "rubric_registry_validation": {
            **rubric_validation,
            "registry_sha256": file_sha256(DEFAULT_RUBRIC_REGISTRY),
        },
        "formal_tokenizer_boundary_probes": boundary_probes,
        "tokenizer_lock_sha256": tokenizer_report["tokenizer_lock"][
            "tokenizer_lock_sha256"
        ],
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
            "independent_reconstruction": report["independent_reconstruction"],
            "locked_train_score_vector_sha256": report[
                "locked_train_score_vector_sha256"
            ],
            "rubric_registry_validation": report["rubric_registry_validation"],
            "formal_tokenizer_boundary_probes": (
                report["formal_tokenizer_boundary_probes"]
            ),
            "tokenizer_lock_sha256": report["tokenizer_lock_sha256"],
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
