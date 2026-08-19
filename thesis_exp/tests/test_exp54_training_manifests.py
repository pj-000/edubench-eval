import json
import copy
from pathlib import Path

import pytest

from thesis_exp.exp54_rar_sft.audit_training_manifests import (
    _period_control_report,
    verify_manifest_row,
)
from thesis_exp.exp54_rar_sft.manifest_source_contract import (
    resolve_expected_arm_source,
)
from thesis_exp.exp54_rar_sft.reference_schedule import schedule_index
from thesis_exp.exp54_rar_sft.training_contract import (
    CONTRACT_VERSION,
    RATIONALE_BOUNDARY_PADDING,
    build_prompt_cache_row,
    materialize_sequence,
    sha256_bytes,
    tokenize_target,
    token_ids_sha256,
)


class CharacterTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [
                (index, index + 1) for index in range(len(text))
            ],
        }

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert enable_thinking is False
        rendered = "".join(
            f"<{item['role']}>\n{item['content']}<end>\n"
            for item in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>\n"
        return [ord(character) for character in rendered] if tokenize else rendered


def _event() -> dict:
    return {
        "event_id": "recipient-event",
        "seed": 42,
        "epoch_index": 1,
        "record_id": "recipient-row",
        "selected_reference_id": "recipient-ref",
        "selected_reason_bytes_sha256": "recipient-hash",
    }


def _prompt_source_row() -> dict:
    row = json.loads(
        Path(
            "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl"
        ).read_text(encoding="utf-8").splitlines()[0]
    )
    return {
        **row,
        "record_id": "manifest-row",
        "question": "Question for the manifest audit?",
        "answer": "Audited answer.",
        "label_5": 4,
    }


def _valid_manifest_fixture():
    tokenizer = CharacterTokenizer()
    source = _prompt_source_row()
    prompt = build_prompt_cache_row(tokenizer, source, row_position=0)
    target = tokenize_target(
        tokenizer,
        score=4,
        rationale="A valid rationale.",
        rationale_active=True,
    )
    sequence = materialize_sequence(tokenizer, prompt, target)
    config = {
        "cutoff_len": 2048,
        "active_rationale_unsupervised_boundary_padding": " ",
    }
    base_event = {
        "event_id": "event-1",
        "seed": 42,
        "epoch_index": 0,
        "epoch_number": 1,
        "row_position": 0,
        "record_id": "manifest-row",
        "label_5": 4,
        "metric_id": source["metric_id"],
        "language": source["language"],
        "selected_reference_id": "aligned-ref",
    }
    expected_source = {
        "rationale": "A valid rationale.",
        "rationale_active": True,
        "arm_selected_reference_id": "aligned-ref",
        "arm_rationale_source_event_id": "event-1",
        "inactive_reason": "",
    }
    row = {
        "contract_version": CONTRACT_VERSION,
        "candidate_status": "CANDIDATE_NOT_FROZEN",
        "arm": "R3",
        "base_event_id": "event-1",
        "seed": 42,
        "epoch_index": 0,
        "epoch_number": 1,
        "row_position": 0,
        "record_id": "manifest-row",
        "prompt_cache_id": prompt["prompt_cache_id"],
        "prompt_token_ids_sha256": prompt["prompt_token_ids_sha256"],
        "score_target": 4,
        "score_loss_active": True,
        "base_selected_reference_id": "aligned-ref",
        "arm_selected_reference_id": "aligned-ref",
        "arm_rationale_source_event_id": "event-1",
        "inactive_reason": "",
        "rationale": "A valid rationale.",
        "rationale_active": True,
        "rationale_boundary_padding": RATIONALE_BOUNDARY_PADDING,
        "rationale_bytes_sha256": sha256_bytes(
            "A valid rationale.".encode("utf-8")
        ),
        **target,
        **sequence,
        "cutoff_len": 2048,
        "padding_mode": "fixed_max_length",
        "padding_token_count": 2048 - sequence["sequence_token_count"],
        "truncated": False,
        "packing": False,
        "score_block_weight": 1.0,
        "rationale_block_weight": 1.0,
    }
    return (
        tokenizer,
        row,
        {prompt["prompt_cache_id"]: prompt},
        source,
        base_event,
        expected_source,
        config,
    )


def _inputs() -> dict:
    return {
        "all_references_by_record": {
            "recipient-row": {
                "references": [
                    {
                        "reference_id": "all-ref-1",
                        "reason": "all reason one",
                    },
                    {
                        "reference_id": "all-ref-2",
                        "reason": "all reason two",
                    },
                    {
                        "reference_id": "all-ref-3",
                        "reason": "all reason three",
                    },
                ]
            }
        },
        "consistent_references_by_id": {
            "recipient-ref": {
                "reference_id": "recipient-ref",
                "reason": "aligned reason",
                "clean_reason_sha256": "recipient-hash",
            },
            "donor-ref": {
                "reference_id": "donor-ref",
                "reason": "shuffled reason",
                "clean_reason_sha256": "donor-hash",
            },
        },
        "donor_by_event": {
            "recipient-event": {
                "active": True,
                "donor_event_id": "donor-event",
                "donor_reference_id": "donor-ref",
                "donor_reason_bytes_sha256": "donor-hash",
            },
            "donor-event": {
                "active": True,
                "recipient_reference_id": "donor-ref",
            },
        },
        "mask_by_event": {
            "recipient-event": {
                "rationale_active": True,
                "inactive_reason": "",
            }
        },
    }


def _select(arm: str, event: dict, inputs: dict) -> dict:
    return resolve_expected_arm_source(
        arm=arm,
        train_row={"record_id": event["record_id"]},
        base_event=event,
        **inputs,
    )


def test_s0_is_score_only_and_r1_uses_shared_all_rater_schedule() -> None:
    event = _event()
    inputs = _inputs()

    s0 = _select("S0", event, inputs)
    r1 = _select("R1", event, inputs)
    expected_index = schedule_index(42, "recipient-row", 1, 3)

    assert s0["rationale"] == ""
    assert s0["rationale_active"] is False
    assert r1["rationale"] == inputs["all_references_by_record"][
        "recipient-row"
    ]["references"][expected_index]["reason"]
    assert r1["rationale_active"] is True
    assert r1["arm_rationale_source_event_id"] == "recipient-event"


def test_r2_uses_donor_event_while_r3_uses_base_selected_reference() -> None:
    event = _event()
    inputs = _inputs()

    r2 = _select("R2", event, inputs)
    r3 = _select("R3", event, inputs)

    assert r2["rationale"] == "shuffled reason"
    assert r2["arm_selected_reference_id"] == "donor-ref"
    assert r2["arm_rationale_source_event_id"] == "donor-event"
    assert r3["rationale"] == "aligned reason"
    assert r3["arm_selected_reference_id"] == "recipient-ref"
    assert r3["arm_rationale_source_event_id"] == "recipient-event"


def test_r2_r3_inactive_event_materializes_empty_rationale_symmetrically() -> None:
    event = _event()
    inputs = _inputs()
    inputs["donor_by_event"]["recipient-event"]["active"] = False
    inputs["mask_by_event"]["recipient-event"] = {
        "rationale_active": False,
        "inactive_reason": "no_legal_strict_event_permutation",
    }

    r2 = _select("R2", event, inputs)
    r3 = _select("R3", event, inputs)

    assert r2 == r3
    assert r2["rationale"] == ""
    assert r2["rationale_active"] is False


def test_donor_mask_activity_mismatch_is_a_hard_failure() -> None:
    inputs = _inputs()
    inputs["mask_by_event"]["recipient-event"]["rationale_active"] = False

    with pytest.raises(ValueError, match="activity differ"):
        _select("R2", _event(), inputs)


def test_independent_auditor_accepts_a_fully_rebuilt_row() -> None:
    (
        tokenizer,
        row,
        prompts,
        train_row,
        base_event,
        expected_source,
        config,
    ) = _valid_manifest_fixture()

    verified = verify_manifest_row(
        tokenizer,
        row,
        prompts,
        train_row,
        base_event,
        expected_source,
        arm="R3",
        seed=42,
        index=0,
        config=config,
    )

    assert verified["_audit_score_mask_token_count"] == 1
    assert verified["_audit_rationale_mask_token_count"] == len(
        "A valid rationale."
    )
    assert (
        verified["_audit_sequence_token_count"]
        + verified["_audit_padding_token_count"]
        == 2048
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "target_text",
        "contract_version",
        "target_bytes_hash",
        "internally_rehashed_target_ids",
        "prompt_cache_id",
        "record_id",
        "row_position",
        "prompt_hash",
        "suffix_ids_and_hash",
        "full_sequence_hash",
        "duplicate_local_positions",
        "noncanonical_absolute_positions",
        "score_token_ids",
        "rationale_token_ids",
        "score_mask_hash",
        "rationale_mask_hash",
        "target_missing_protocol_space",
        "target_double_protocol_space",
        "missing_boundary_padding",
        "double_boundary_padding",
        "inactive_nonempty_rationale",
    ],
)
def test_independent_auditor_rejects_manifest_tampering(tamper) -> None:
    (
        tokenizer,
        original,
        prompts,
        train_row,
        base_event,
        expected_source,
        config,
    ) = _valid_manifest_fixture()
    row = copy.deepcopy(original)
    if tamper == "target_text":
        row["target_text"] = row["target_text"].replace("valid", "altered")
    elif tamper == "contract_version":
        row["contract_version"] = "another-contract"
    elif tamper == "target_bytes_hash":
        row["target_bytes_sha256"] = "0" * 64
    elif tamper == "internally_rehashed_target_ids":
        row["target_token_ids"][0] += 1
        row["target_token_ids_sha256"] = token_ids_sha256(
            row["target_token_ids"]
        )
    elif tamper == "prompt_cache_id":
        row["prompt_cache_id"] = "prompt-missing"
    elif tamper == "record_id":
        row["record_id"] = "another-record"
    elif tamper == "row_position":
        row["row_position"] = 1
    elif tamper == "prompt_hash":
        row["prompt_token_ids_sha256"] = "1" * 64
    elif tamper == "suffix_ids_and_hash":
        row["assistant_suffix_token_ids"][0] += 1
        row["assistant_suffix_token_ids_sha256"] = token_ids_sha256(
            row["assistant_suffix_token_ids"]
        )
    elif tamper == "full_sequence_hash":
        row["full_token_ids_sha256"] = "2" * 64
    elif tamper == "duplicate_local_positions":
        row["score_token_positions_in_target"] *= 2
    elif tamper == "noncanonical_absolute_positions":
        row["rationale_token_positions"] = list(
            reversed(row["rationale_token_positions"])
        )
    elif tamper == "score_token_ids":
        row["score_token_ids"][0] += 1
    elif tamper == "rationale_token_ids":
        row["rationale_token_ids"][0] += 1
    elif tamper == "score_mask_hash":
        row["score_mask_sha256"] = "3" * 64
    elif tamper == "rationale_mask_hash":
        row["rationale_mask_sha256"] = "4" * 64
    elif tamper == "target_missing_protocol_space":
        row["target_text"] = row["target_text"].replace('. "}', '."}')
        row["target_bytes_sha256"] = sha256_bytes(
            row["target_text"].encode("utf-8")
        )
    elif tamper == "target_double_protocol_space":
        row["target_text"] = row["target_text"].replace('. "}', '.  "}')
        row["target_bytes_sha256"] = sha256_bytes(
            row["target_text"].encode("utf-8")
        )
    elif tamper == "missing_boundary_padding":
        row["rationale_boundary_padding"] = ""
    elif tamper == "double_boundary_padding":
        row["rationale_boundary_padding"] = "  "
    elif tamper == "inactive_nonempty_rationale":
        row["rationale_active"] = False
        row["rationale_boundary_padding"] = ""
        row["rationale_block_weight"] = 0.0
    else:
        raise AssertionError(tamper)

    with pytest.raises((ValueError, AssertionError)):
        verify_manifest_row(
            tokenizer,
            row,
            prompts,
            train_row,
            base_event,
            expected_source,
            arm="R3",
            seed=42,
            index=0,
            config=config,
        )


def _rematerialize_row(
    tokenizer,
    row,
    prompt,
    *,
    score,
    rationale,
    rationale_active,
) -> None:
    target = tokenize_target(
        tokenizer,
        score=score,
        rationale=rationale,
        rationale_active=rationale_active,
    )
    sequence = materialize_sequence(tokenizer, prompt, target)
    row.update(target)
    row.update(sequence)
    row["score_target"] = score
    row["rationale"] = rationale
    row["rationale_active"] = rationale_active
    row["rationale_boundary_padding"] = (
        RATIONALE_BOUNDARY_PADDING if rationale_active else ""
    )
    row["rationale_bytes_sha256"] = sha256_bytes(rationale.encode("utf-8"))
    row["rationale_block_weight"] = 1.0 if rationale_active else 0.0
    row["padding_token_count"] = 2048 - sequence["sequence_token_count"]


def _semantic_fixture():
    (
        tokenizer,
        row,
        prompts,
        train_row,
        base_event,
        expected_source,
        config,
    ) = _valid_manifest_fixture()
    prompt = prompts[row["prompt_cache_id"]]
    return (
        tokenizer,
        row,
        prompts,
        prompt,
        train_row,
        base_event,
        expected_source,
        config,
    )


def test_rebuilt_score_target_must_equal_locked_train_label() -> None:
    (
        tokenizer,
        row,
        prompts,
        prompt,
        train_row,
        base_event,
        expected_source,
        config,
    ) = _semantic_fixture()
    _rematerialize_row(
        tokenizer,
        row,
        prompt,
        score=3,
        rationale=row["rationale"],
        rationale_active=True,
    )

    with pytest.raises(ValueError, match="locked label_5"):
        verify_manifest_row(
            tokenizer,
            row,
            prompts,
            train_row,
            base_event,
            expected_source,
            arm="R3",
            seed=42,
            index=0,
            config=config,
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "swapped_r3_rationale",
        "r3_reference_and_rationale",
        "alternate_r2_donor_edge",
        "r2_donor_event_only",
        "wrong_r1_reference",
        "rationale_activity",
        "inactive_reason",
        "source_ids",
        "base_event_id",
    ],
)
def test_rebuilt_semantic_source_tampering_is_rejected(tamper) -> None:
    (
        tokenizer,
        row,
        prompts,
        prompt,
        train_row,
        base_event,
        expected_source,
        config,
    ) = _semantic_fixture()
    arm = "R3"
    if tamper == "swapped_r3_rationale":
        _rematerialize_row(
            tokenizer,
            row,
            prompt,
            score=4,
            rationale="Rationale from another R3 event.",
            rationale_active=True,
        )
    elif tamper == "r3_reference_and_rationale":
        row["arm_selected_reference_id"] = "another-aligned-reference"
        _rematerialize_row(
            tokenizer,
            row,
            prompt,
            score=4,
            rationale="Another internally consistent rationale.",
            rationale_active=True,
        )
    elif tamper == "alternate_r2_donor_edge":
        arm = "R2"
        expected_source = {
            "rationale": "Frozen donor rationale.",
            "rationale_active": True,
            "arm_selected_reference_id": "frozen-donor-reference",
            "arm_rationale_source_event_id": "frozen-donor-event",
            "inactive_reason": "",
        }
        row["arm"] = "R2"
        row["arm_selected_reference_id"] = "frozen-donor-reference"
        row["arm_rationale_source_event_id"] = "frozen-donor-event"
        _rematerialize_row(
            tokenizer,
            row,
            prompt,
            score=4,
            rationale="Frozen donor rationale.",
            rationale_active=True,
        )
        row["arm_selected_reference_id"] = "alternate-donor-reference"
        row["arm_rationale_source_event_id"] = "alternate-donor-event"
        _rematerialize_row(
            tokenizer,
            row,
            prompt,
            score=4,
            rationale="Alternate legal but unfrozen donor rationale.",
            rationale_active=True,
        )
    elif tamper == "r2_donor_event_only":
        arm = "R2"
        expected_source = {
            "rationale": "Frozen donor rationale.",
            "rationale_active": True,
            "arm_selected_reference_id": "frozen-donor-reference",
            "arm_rationale_source_event_id": "frozen-donor-event",
            "inactive_reason": "",
        }
        row["arm"] = "R2"
        row["arm_selected_reference_id"] = "frozen-donor-reference"
        _rematerialize_row(
            tokenizer,
            row,
            prompt,
            score=4,
            rationale="Frozen donor rationale.",
            rationale_active=True,
        )
        row["arm_rationale_source_event_id"] = "another-donor-event"
    elif tamper == "wrong_r1_reference":
        arm = "R1"
        expected_source = {
            "rationale": "Scheduled R1 rationale.",
            "rationale_active": True,
            "arm_selected_reference_id": "scheduled-all-rater-reference",
            "arm_rationale_source_event_id": "event-1",
            "inactive_reason": "",
        }
        row["arm"] = "R1"
        row["arm_selected_reference_id"] = "scheduled-all-rater-reference"
        _rematerialize_row(
            tokenizer,
            row,
            prompt,
            score=4,
            rationale="Scheduled R1 rationale.",
            rationale_active=True,
        )
        row["arm_selected_reference_id"] = "unscheduled-all-rater-reference"
    elif tamper == "rationale_activity":
        row["arm_selected_reference_id"] = None
        row["arm_rationale_source_event_id"] = None
        row["inactive_reason"] = "tampered_inactive"
        _rematerialize_row(
            tokenizer,
            row,
            prompt,
            score=4,
            rationale="",
            rationale_active=False,
        )
    elif tamper == "inactive_reason":
        row["inactive_reason"] = "another_reason"
    elif tamper == "source_ids":
        row["arm_rationale_source_event_id"] = "another-source-event"
    elif tamper == "base_event_id":
        row["base_event_id"] = "new-unique-base-event"
    else:
        raise AssertionError(tamper)

    with pytest.raises(ValueError, match="source differs|metadata differs"):
        verify_manifest_row(
            tokenizer,
            row,
            prompts,
            train_row,
            base_event,
            expected_source,
            arm=arm,
            seed=42,
            index=0,
            config=config,
        )


def test_all_event_target_counter_detects_inactive_target_tamper() -> None:
    common = {
        "rationale_active": False,
        "rationale_bytes_sha256": sha256_bytes(b""),
        "rationale_token_ids": [],
        "_audit_rationale_mask_token_count": 0,
        "_audit_sequence_token_count": 100,
        "_audit_padding_token_count": 1948,
    }
    r2 = [
        {
            **common,
            "target_bytes_sha256": "a" * 64,
            "target_token_ids_sha256": "b" * 64,
        }
    ]
    r3 = [
        {
            **common,
            "target_bytes_sha256": "c" * 64,
            "target_token_ids_sha256": "d" * 64,
        }
    ]

    with pytest.raises(AssertionError, match="all_event_target"):
        _period_control_report(r2, r3)


def test_formal_materialized_candidate_report_is_aggregate_and_not_ready() -> None:
    base = Path("thesis_exp/outputs/exp54_rar_sft/rar_v2")
    report_path = base / "audit/materialized_manifest_candidate_report.json"
    lock_path = base / "protocol/materialized_manifest_candidate_lock.json"
    if not report_path.exists() or not lock_path.exists():
        pytest.skip("formal materialized-manifest report is unavailable")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert report["status"] == (
        "MATERIALIZED_MANIFEST_CANDIDATE_AUDITED_NOT_FROZEN"
    )
    assert lock["status"] == report["status"]
    for artifact in (report, lock):
        assert artifact["manifest_freeze_allowed"] is False
        assert artifact["smoke_training_allowed"] is False
        assert artifact["formal_training_allowed"] is False
        assert artifact["dev_accessed"] is False
        assert artifact["test_accessed"] is False
        assert artifact["training_used"] is False

    assert [seed["seed"] for seed in report["seed_reports"]] == [42, 43, 44]
    assert all(report["independent_reconstruction"].values())
    rubric = report["rubric_registry_validation"]
    assert rubric["train_rows_validated"] == 2_654
    assert rubric["metric_count"] == rubric["expected_metric_count"] == 12
    assert (
        rubric["metric_language_entry_count"]
        == rubric["expected_metric_language_entry_count"]
        == 24
    )
    assert rubric["all_rows_match_canonical_registry"] is True
    assert all(
        probe["boundary_validation_passed"]
        for probe in report["formal_tokenizer_boundary_probes"].values()
    )
    assert {
        "prompt_cleaning_source",
        "shared_reference_schedule",
        "canonical_rubric_registry",
        "manifest_source_contract",
    }.issubset(report["source_hashes"])
    assert report["independent_reconstruction"][
        "score_targets_bound_to_locked_train_labels"
    ] is True
    assert report["independent_reconstruction"][
        "arm_sources_bound_to_locked_reference_schedule_donor_mask"
    ] is True
    for seed in report["seed_reports"]:
        assert all(seed["checks"].values())
        assert seed["score_targets_equal_locked_train"] is True
        assert (
            seed["locked_train_score_vector_sha256"]
            == report["locked_train_score_vector_sha256"]
        )
        assert set(seed["score_target_vector_sha256_by_arm"].values()) == {
            report["locked_train_score_vector_sha256"]
        }
        source_audit = seed["semantic_source_audit"]
        assert source_audit["base_schedule_rows_verified"] == 7_962
        assert set(source_audit["arm_source_rows_verified"].values()) == {
            7_962
        }
        assert source_audit["r1_schedule_source_mismatches"] == 0
        assert source_audit["r2_donor_backlink_mismatches"] == 0
        assert source_audit["r3_aligned_reference_mismatches"] == 0
        assert source_audit["r2_r3_mask_mismatches"] == 0
        assert set(
            source_audit["expected_source_vector_sha256_by_arm"]
        ) == {"S0", "R1", "R2", "R3"}
        budget = seed["training_budget"]
        assert all(budget["checks"].values())
        assert budget["optimizer_steps_total"] == 996
        padded = {
            arm: values["fixed_padded_input_tokens"]
            for arm, values in budget["per_arm"].items()
        }
        assert set(padded.values()) == {16_306_176}
        for values in budget["per_arm"].values():
            assert values["score_supervised_events"] == 7_962
            assert values["maximum_sequence_tokens"] <= 2_048
            assert values["collator_totals_match_independent_rebuild"] is True
            assert (
                values["collator_boolean_score_mask_tokens"]
                == values["score_supervised_tokens"]
            )
            assert (
                values["collator_boolean_rationale_mask_tokens"]
                == values["rationale_supervised_tokens"]
            )
            assert (
                values["collator_fixed_padded_tokens"]
                == values["fixed_padded_input_tokens"]
            )
        assert (
            budget["per_arm"]["R2"]["unpadded_sequence_tokens"]
            == budget["per_arm"]["R3"]["unpadded_sequence_tokens"]
        )
        assert (
            budget["per_arm"]["R2"]["rationale_supervised_events"]
            == budget["per_arm"]["R3"]["rationale_supervised_events"]
            == 4_803
        )
        for epoch in seed["r2_r3_frequency_control"]["individual_epochs"]:
            assert epoch["active_rationale_events"] == 1_601
            assert not any(epoch["frequency_l1_difference"].values())
            assert all(epoch["checks"].values())
            assert (
                epoch["rationale_active_vector_sha256_R2"]
                == epoch["rationale_active_vector_sha256_R3"]
            )
            assert (
                epoch["unpadded_sequence_token_total_R2"]
                == epoch["unpadded_sequence_token_total_R3"]
            )
            assert (
                epoch["fixed_padded_token_total_R2"]
                == epoch["fixed_padded_token_total_R3"]
            )

    forbidden_public_keys = {
        "record_id",
        "reference_id",
        "base_event_id",
        "target_text",
        "reason",
        "rationale",
    }

    def collect_keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(
                *(collect_keys(item) for item in value.values())
            )
        if isinstance(value, list):
            return set().union(*(collect_keys(item) for item in value))
        return set()

    assert not (collect_keys(report) & forbidden_public_keys)
    assert not (collect_keys(lock) & forbidden_public_keys)
