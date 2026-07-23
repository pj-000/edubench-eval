"""Build private candidate S0/R1/R2/R3 materialized training manifests.

The builder consumes only the frozen train split, private reference/event
artifacts, and the locked tokenizer. It never reads dev/test, freezes a
manifest, authorizes training, or starts a trainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import (
    file_sha256,
    read_jsonl,
    reject_eval_path,
    write_jsonl,
)
from thesis_exp.exp54_rar_sft.reference_schedule import (
    EXPECTED_TRAIN_ROWS,
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
    schedule_index,
)
from thesis_exp.exp54_rar_sft.training_contract import (
    CONTRACT_VERSION,
    RATIONALE_BOUNDARY_PADDING,
    build_prompt_cache_row,
    materialize_sequence,
    sha256_bytes,
    tokenize_target,
)


ARMS = ("S0", "R1", "R2", "R3")
EXPECTED_EVENTS_PER_SEED = EXPECTED_TRAIN_ROWS * FORMAL_EPOCHS
DEFAULT_TRAIN = (
    REPO_ROOT
    / "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
)
DEFAULT_ALL_REFERENCES = (
    DEFAULT_OUTPUT / "data/all_rater_reference_sets.jsonl"
)
DEFAULT_CONSISTENT_REFERENCES = (
    DEFAULT_OUTPUT / "data/label_consistent_reference_sets.jsonl"
)
DEFAULT_REFERENCE_LOCK = (
    DEFAULT_OUTPUT / "protocol/reference_set_data_lock.json"
)
DEFAULT_SCHEDULE_LOCK = (
    DEFAULT_OUTPUT / "protocol/reference_schedule_candidate_lock.json"
)
DEFAULT_DONOR_LOCK = (
    DEFAULT_OUTPUT / "protocol/r2_event_donor_map_candidate_lock.json"
)
DEFAULT_MASK_LOCK = (
    DEFAULT_OUTPUT / "protocol/r2_r3_event_mask_candidate_lock.json"
)
DEFAULT_TOKENIZER_REPORT = (
    DEFAULT_OUTPUT / "audit/r2_event_donor_match_report_seed42.json"
)


def _index_unique(
    rows: list[dict[str, Any]],
    key: str,
    *,
    source: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            raise ValueError(f"{source}: empty {key}")
        if value in output:
            raise ValueError(f"{source}: duplicate {key}: {value}")
        output[value] = row
    return output


def _reference_indexes(
    rows: list[dict[str, Any]],
    *,
    source: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_record = _index_unique(rows, "record_id", source=source)
    by_reference: dict[str, dict[str, Any]] = {}
    for record_id, row in by_record.items():
        references = list(row.get("references") or [])
        if int(row["reference_count"]) != len(references):
            raise ValueError(f"{record_id}: declared reference_count differs")
        seen_raters: set[str] = set()
        for reference in references:
            reference_id = str(reference.get("reference_id") or "")
            rater_id = str(reference.get("rater_id") or "")
            reason = str(reference.get("reason") or "")
            if not reference_id or not rater_id or not reason:
                raise ValueError(f"{record_id}: incomplete reference")
            if rater_id in seen_raters:
                raise ValueError(f"{record_id}: duplicate rater_id {rater_id}")
            if reference_id in by_reference:
                raise ValueError(f"{source}: duplicate reference_id {reference_id}")
            seen_raters.add(rater_id)
            expected_hash = str(reference.get("clean_reason_sha256") or "")
            actual_hash = sha256_bytes(reason.encode("utf-8"))
            if expected_hash != actual_hash:
                raise ValueError(f"{reference_id}: clean reason hash differs")
            by_reference[reference_id] = {
                **reference,
                "record_id": record_id,
                "label_5": int(row["label_5"]),
                "metric_id": str(row["metric_id"]),
                "language": str(row["language"]),
                "normalized_qa_key": str(row["normalized_qa_key"]),
            }
    return by_record, by_reference


def build_prompt_cache(
    tokenizer: Any,
    train_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(train_rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"expected {EXPECTED_TRAIN_ROWS} train rows, found {len(train_rows)}"
        )
    cache = [
        build_prompt_cache_row(tokenizer, row, row_position=row_position)
        for row_position, row in enumerate(train_rows)
    ]
    if len({row["record_id"] for row in cache}) != len(cache):
        raise ValueError("train rows contain duplicate record_id")
    return cache


def _validate_schedule(
    events: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> None:
    if len(events) != EXPECTED_EVENTS_PER_SEED:
        raise ValueError(
            f"seed {seed}: expected {EXPECTED_EVENTS_PER_SEED} events, "
            f"found {len(events)}"
        )
    if len({str(event["event_id"]) for event in events}) != len(events):
        raise ValueError(f"seed {seed}: duplicate base event ID")
    for event_index, event in enumerate(events):
        epoch_index, row_position = divmod(event_index, EXPECTED_TRAIN_ROWS)
        train_row = train_rows[row_position]
        expected = {
            "seed": seed,
            "epoch_index": epoch_index,
            "epoch_number": epoch_index + 1,
            "row_position": row_position,
            "record_id": str(train_row["record_id"]),
            "label_5": int(train_row["label_5"]),
            "metric_id": str(train_row["metric_id"]),
            "language": str(train_row["language"]),
        }
        for field, value in expected.items():
            if event[field] != value:
                raise ValueError(
                    f"seed {seed}/event {event_index}: {field} differs "
                    "from train order"
                )


def _validate_mask(
    events: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, dict[str, Any]]:
    if len(mask_rows) != len(events):
        raise ValueError(f"seed {seed}: event mask length differs")
    mask_by_event = _index_unique(
        mask_rows,
        "base_event_id",
        source=f"seed {seed} event mask",
    )
    for event, mask in zip(events, mask_rows, strict=True):
        for event_field, mask_field in (
            ("event_id", "base_event_id"),
            ("record_id", "record_id"),
            ("epoch_index", "epoch_index"),
            ("epoch_number", "epoch_number"),
            ("row_position", "row_position"),
            ("seed", "seed"),
        ):
            if event[event_field] != mask[mask_field]:
                raise ValueError(
                    f"seed {seed}/{event['event_id']}: mask {mask_field} differs"
                )
        forbidden = {"score_active", "score_mask", "score_loss_active"} & set(mask)
        if forbidden:
            raise ValueError(f"event mask contains score deactivation: {forbidden}")
    return mask_by_event


def _validate_donor_map(
    events: list[dict[str, Any]],
    donor_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, dict[str, Any]]:
    if len(donor_rows) != len(events):
        raise ValueError(f"seed {seed}: donor map length differs")
    donor_by_event = _index_unique(
        donor_rows,
        "recipient_event_id",
        source=f"seed {seed} donor map",
    )
    if set(donor_by_event) != {str(event["event_id"]) for event in events}:
        raise ValueError(f"seed {seed}: donor map does not close over base events")
    return donor_by_event


def select_arm_rationale(
    *,
    arm: str,
    event: dict[str, Any],
    all_references_by_record: dict[str, dict[str, Any]],
    consistent_references_by_id: dict[str, dict[str, Any]],
    donor_by_event: dict[str, dict[str, Any]],
    mask_by_event: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    event_id = str(event["event_id"])
    record_id = str(event["record_id"])
    if arm == "S0":
        return {
            "rationale": "",
            "rationale_active": False,
            "arm_selected_reference_id": None,
            "arm_rationale_source_event_id": None,
            "inactive_reason": "score_only_arm",
        }

    if arm == "R1":
        reference_row = all_references_by_record[record_id]
        references = list(reference_row.get("references") or [])
        if not references:
            return {
                "rationale": "",
                "rationale_active": False,
                "arm_selected_reference_id": None,
                "arm_rationale_source_event_id": None,
                "inactive_reason": "no_human_reference",
            }
        reference = references[
            schedule_index(
                int(event["seed"]),
                record_id,
                int(event["epoch_index"]),
                len(references),
            )
        ]
        return {
            "rationale": str(reference["reason"]),
            "rationale_active": True,
            "arm_selected_reference_id": str(reference["reference_id"]),
            "arm_rationale_source_event_id": event_id,
            "inactive_reason": "",
        }

    mask = mask_by_event[event_id]
    donor = donor_by_event[event_id]
    active = bool(mask["rationale_active"])
    if bool(donor["active"]) != active:
        raise ValueError(f"{event_id}: donor and R2/R3 mask activity differ")
    if not active:
        return {
            "rationale": "",
            "rationale_active": False,
            "arm_selected_reference_id": None,
            "arm_rationale_source_event_id": None,
            "inactive_reason": str(mask["inactive_reason"]),
        }

    reference_id = (
        str(donor["donor_reference_id"])
        if arm == "R2"
        else str(event["selected_reference_id"])
    )
    source_event_id = (
        str(donor["donor_event_id"]) if arm == "R2" else event_id
    )
    reference = consistent_references_by_id.get(reference_id)
    if reference is None:
        raise ValueError(f"{event_id}: missing {arm} reference {reference_id}")
    reason_hash_field = (
        "donor_reason_bytes_sha256"
        if arm == "R2"
        else "selected_reason_bytes_sha256"
    )
    expected_reason_hash = (
        str(donor[reason_hash_field])
        if arm == "R2"
        else str(event[reason_hash_field])
    )
    if expected_reason_hash != reference["clean_reason_sha256"]:
        raise ValueError(f"{event_id}: {arm} reason hash differs from source")
    if arm == "R2":
        source_event = donor_by_event.get(source_event_id)
        if source_event is None:
            raise ValueError(f"{event_id}: donor source event is outside closure")
        if str(source_event["recipient_reference_id"]) != reference_id:
            raise ValueError(f"{event_id}: donor event/reference backlink differs")
    return {
        "rationale": str(reference["reason"]),
        "rationale_active": True,
        "arm_selected_reference_id": reference_id,
        "arm_rationale_source_event_id": source_event_id,
        "inactive_reason": "",
    }


def materialize_arm(
    tokenizer: Any,
    *,
    arm: str,
    seed: int,
    train_rows: list[dict[str, Any]],
    prompt_cache: list[dict[str, Any]],
    events: list[dict[str, Any]],
    all_references_by_record: dict[str, dict[str, Any]],
    consistent_references_by_id: dict[str, dict[str, Any]],
    donor_by_event: dict[str, dict[str, Any]],
    mask_by_event: dict[str, dict[str, Any]],
    cutoff_len: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    over_cutoff: list[tuple[str, int]] = []
    for event in events:
        row_position = int(event["row_position"])
        train_row = train_rows[row_position]
        prompt = prompt_cache[row_position]
        selected = select_arm_rationale(
            arm=arm,
            event=event,
            all_references_by_record=all_references_by_record,
            consistent_references_by_id=consistent_references_by_id,
            donor_by_event=donor_by_event,
            mask_by_event=mask_by_event,
        )
        target = tokenize_target(
            tokenizer,
            score=int(train_row["label_5"]),
            rationale=selected["rationale"],
            rationale_active=selected["rationale_active"],
        )
        sequence = materialize_sequence(tokenizer, prompt, target)
        sequence_length = int(sequence["sequence_token_count"])
        if sequence_length > cutoff_len:
            over_cutoff.append((str(event["event_id"]), sequence_length))
        rows.append(
            {
                "contract_version": CONTRACT_VERSION,
                "candidate_status": "CANDIDATE_NOT_FROZEN",
                "arm": arm,
                "base_event_id": str(event["event_id"]),
                "seed": seed,
                "epoch_index": int(event["epoch_index"]),
                "epoch_number": int(event["epoch_number"]),
                "row_position": row_position,
                "record_id": str(event["record_id"]),
                "prompt_cache_id": str(prompt["prompt_cache_id"]),
                "prompt_token_ids_sha256": str(prompt["prompt_token_ids_sha256"]),
                "score_target": int(train_row["label_5"]),
                "score_loss_active": True,
                "base_selected_reference_id": event["selected_reference_id"],
                **selected,
                "rationale_bytes_sha256": sha256_bytes(
                    selected["rationale"].encode("utf-8")
                ),
                "rationale_boundary_padding": (
                    RATIONALE_BOUNDARY_PADDING
                    if selected["rationale_active"]
                    else ""
                ),
                **target,
                **sequence,
                "cutoff_len": cutoff_len,
                "padding_mode": "fixed_max_length",
                "padding_token_count": cutoff_len - sequence_length,
                "truncated": False,
                "packing": False,
                "score_block_weight": 1.0,
                "rationale_block_weight": (
                    1.0 if selected["rationale_active"] else 0.0
                ),
            }
        )
    if over_cutoff:
        longest = sorted(over_cutoff, key=lambda item: item[1], reverse=True)[:10]
        raise ValueError(
            f"seed {seed}/{arm}: {len(over_cutoff)} sequences exceed "
            f"cutoff_len={cutoff_len}; longest={longest}"
        )
    return rows


def _load_locked_tokenizer(tokenizer_path: Path, tokenizer_report: dict[str, Any]):
    from transformers import AutoTokenizer

    tokenizer_lock = tokenizer_report["tokenizer_lock"]
    if tokenizer_lock.get("status") != "QWEN_TOKENIZER_REVISION_LOCKED":
        raise ValueError("tokenizer report is not formally locked")
    expected_files = {
        str(item["path"]): str(item["sha256"])
        for item in tokenizer_lock["tokenizer_files"]
    }
    for filename, expected_hash in expected_files.items():
        path = tokenizer_path / filename
        if not path.exists() or file_sha256(path) != expected_hash:
            raise ValueError(f"tokenizer file differs from lock: {filename}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.__class__.__name__ != tokenizer_lock["tokenizer_class"]:
        raise ValueError("tokenizer class differs from lock")
    if len(tokenizer) != int(tokenizer_lock["vocab_size"]):
        raise ValueError("tokenizer vocabulary size differs from lock")
    return tokenizer


def _require_hash(path: Path, expected_hash: str, *, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if file_sha256(path) != expected_hash:
        raise ValueError(f"{name} differs from its candidate lock")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--all-references", type=Path, default=DEFAULT_ALL_REFERENCES)
    parser.add_argument(
        "--consistent-references",
        type=Path,
        default=DEFAULT_CONSISTENT_REFERENCES,
    )
    parser.add_argument("--reference-lock", type=Path, default=DEFAULT_REFERENCE_LOCK)
    parser.add_argument("--schedule-lock", type=Path, default=DEFAULT_SCHEDULE_LOCK)
    parser.add_argument("--donor-lock", type=Path, default=DEFAULT_DONOR_LOCK)
    parser.add_argument("--mask-lock", type=Path, default=DEFAULT_MASK_LOCK)
    parser.add_argument(
        "--tokenizer-report",
        type=Path,
        default=DEFAULT_TOKENIZER_REPORT,
    )
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cutoff-len", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cutoff_len < 1:
        raise ValueError("cutoff_len must be positive")
    inputs = (
        args.train,
        args.all_references,
        args.consistent_references,
        args.reference_lock,
        args.schedule_lock,
        args.donor_lock,
        args.mask_lock,
        args.tokenizer_report,
    )
    for path in inputs:
        reject_eval_path(path)
        if not path.exists():
            raise FileNotFoundError(path)

    reference_lock = json.loads(args.reference_lock.read_text(encoding="utf-8"))
    schedule_lock = json.loads(args.schedule_lock.read_text(encoding="utf-8"))
    donor_lock = json.loads(args.donor_lock.read_text(encoding="utf-8"))
    mask_lock = json.loads(args.mask_lock.read_text(encoding="utf-8"))
    tokenizer_report = json.loads(args.tokenizer_report.read_text(encoding="utf-8"))
    for lock_name, lock in (
        ("reference", reference_lock),
        ("schedule", schedule_lock),
        ("donor", donor_lock),
        ("mask", mask_lock),
        ("tokenizer report", tokenizer_report),
    ):
        if lock.get("dev_accessed") or lock.get("test_accessed"):
            raise PermissionError(f"{lock_name} indicates evaluation data access")
        if lock.get("training_used"):
            raise PermissionError(f"{lock_name} indicates prior training use")

    _require_hash(
        args.train,
        str(reference_lock["input"]["rar0_source_hashes"]["train"]),
        name="train split",
    )
    _require_hash(
        args.all_references,
        str(reference_lock["output_hashes"]["all_rater_reference_sets"]),
        name="all-rater reference inventory",
    )
    _require_hash(
        args.consistent_references,
        str(reference_lock["output_hashes"]["label_consistent_reference_sets"]),
        name="label-consistent reference inventory",
    )
    tokenizer = _load_locked_tokenizer(args.tokenizer_path, tokenizer_report)

    train_rows = read_jsonl(args.train, protect_split=True)
    all_reference_rows = read_jsonl(args.all_references, protect_split=True)
    consistent_reference_rows = read_jsonl(
        args.consistent_references,
        protect_split=True,
    )
    all_by_record, _all_by_reference = _reference_indexes(
        all_reference_rows,
        source="all-rater references",
    )
    consistent_by_record, consistent_by_reference = _reference_indexes(
        consistent_reference_rows,
        source="label-consistent references",
    )
    train_ids = [str(row["record_id"]) for row in train_rows]
    if (
        train_ids != list(all_by_record)
        or train_ids != list(consistent_by_record)
    ):
        raise ValueError("train and reference inventories differ in row order")

    prompt_cache = build_prompt_cache(tokenizer, train_rows)
    prompt_cache_path = args.output_dir / "data/shared_prompt_cache.jsonl"
    write_jsonl(prompt_cache_path, prompt_cache)

    for seed in FORMAL_SEEDS:
        schedule_path = (
            args.output_dir / "data" / f"base_event_schedule_seed{seed}.jsonl"
        )
        donor_path = (
            args.output_dir / "data" / f"r2_event_donor_map_seed{seed}.jsonl"
        )
        r2_mask_path = (
            args.output_dir / "data" / f"r2_event_active_mask_seed{seed}.jsonl"
        )
        r3_mask_path = (
            args.output_dir / "data" / f"r3_event_active_mask_seed{seed}.jsonl"
        )
        for path in (schedule_path, donor_path, r2_mask_path, r3_mask_path):
            reject_eval_path(path)
        _require_hash(
            schedule_path,
            str(
                schedule_lock["schedule_file_hashes"][
                    f"base_event_schedule_seed{seed}"
                ]
            ),
            name=f"seed {seed} base schedule",
        )
        _require_hash(
            donor_path,
            str(donor_lock["event_donor_map_sha256_by_seed"][f"seed{seed}"]),
            name=f"seed {seed} event donor map",
        )
        expected_masks = mask_lock["event_mask_sha256_by_seed"][f"seed{seed}"]
        _require_hash(
            r2_mask_path,
            str(expected_masks["r2"]),
            name=f"seed {seed} R2 event mask",
        )
        _require_hash(
            r3_mask_path,
            str(expected_masks["r3"]),
            name=f"seed {seed} R3 event mask",
        )
        if r2_mask_path.read_bytes() != r3_mask_path.read_bytes():
            raise ValueError(f"seed {seed}: R2/R3 masks are not byte-identical")

        events = read_jsonl(schedule_path, protect_split=True)
        donor_rows = read_jsonl(donor_path, protect_split=True)
        mask_rows = read_jsonl(r2_mask_path, protect_split=True)
        _validate_schedule(events, train_rows, seed=seed)
        donor_by_event = _validate_donor_map(events, donor_rows, seed=seed)
        mask_by_event = _validate_mask(events, mask_rows, seed=seed)

        for arm in ARMS:
            manifest = materialize_arm(
                tokenizer,
                arm=arm,
                seed=seed,
                train_rows=train_rows,
                prompt_cache=prompt_cache,
                events=events,
                all_references_by_record=all_by_record,
                consistent_references_by_id=consistent_by_reference,
                donor_by_event=donor_by_event,
                mask_by_event=mask_by_event,
                cutoff_len=args.cutoff_len,
            )
            manifest_path = (
                args.output_dir
                / "data"
                / f"training_manifest_{arm.lower()}_seed{seed}.jsonl"
            )
            write_jsonl(manifest_path, manifest)


if __name__ == "__main__":
    main()
