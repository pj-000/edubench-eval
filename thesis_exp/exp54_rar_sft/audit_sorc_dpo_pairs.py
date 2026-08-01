"""Independently audit private SORC-DPO candidate pair manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import (
    SEEDS,
    any_explicit_score_leakage,
    load_train_rows,
    read_jsonl,
    score_leakage,
    sha256_file,
)


RAR_ROOT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
DATA_ROOT = RAR_ROOT / "data"
OUTPUT_ROOT = RAR_ROOT / "preference_pairs"
FAILURE_ROOT = RAR_ROOT / "actual_failure_bank"
FAILURE_PATH = FAILURE_ROOT / "private/actual_failure_bank.jsonl"
FROZEN_MANIFEST_LOCK = (
    RAR_ROOT / "protocol/materialized_manifest_frozen_lock.json"
)
PROTOCOL_PATH = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "sorc_dpo_pair_protocol_v1.json"
)
TRAINING_CONFIG_PATH = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "training_configuration_candidate.json"
)
PAIR_SCHEMA_VERSION = "exp54-sorc-dpo-pair-v1"
BLOCKS = ("adjacent_score", "severe_l2h", "h2l_guard")


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(*items: Any) -> str:
    return hashlib.sha256(
        "|".join(str(item) for item in items).encode("utf-8")
    ).hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", value),
    ).strip()


def independent_ordinal_cost(gold: int, rejected: int) -> float:
    if gold not in range(1, 6) or rejected not in range(1, 6):
        raise ValueError("score outside 1-5")
    return (
        abs(rejected - gold) / 4.0
        + int(gold <= 2 and rejected >= 4)
    )


def independent_odpo_offset(gold: int, rejected: int) -> float:
    return independent_ordinal_cost(gold, rejected) / 2.0


def independent_synthetic_score(
    pair_type: str,
    record_id: str,
    gold: int,
) -> int:
    if pair_type == "severe_l2h":
        return 4
    if pair_type == "h2l_guard":
        return 2
    if pair_type != "adjacent_score":
        raise ValueError("unknown score block")
    if gold == 1:
        return 2
    if gold == 5:
        return 4
    direction = (
        1
        if int(stable_hash("synthetic-adjacent", record_id), 16) % 2
        else -1
    )
    return gold + direction


def _pair_payload_hash(row: dict[str, Any], field: str) -> str:
    payload = {key: value for key, value in row.items() if key != field}
    return sha256_text(compact_json(payload))


def raw_source_key(digest: str, seed: int) -> tuple[str, int]:
    """Disambiguate byte-identical generations from different checkpoints."""
    if seed not in SEEDS:
        raise ValueError("raw-generation seed differs")
    return digest, seed


def _index_failure_sources() -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
]:
    failure_index: dict[str, dict[str, Any]] = {}
    failure_by_record_seed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_jsonl(FAILURE_PATH):
        digest = sha256_text(compact_json(row))
        if digest in failure_index:
            raise ValueError("duplicate failure evidence hash")
        failure_index[digest] = row
        key = (str(row["record_id"]), int(row["generator_seed"]))
        if key in failure_by_record_seed:
            raise ValueError("duplicate failure record/seed")
        failure_by_record_seed[key] = row
    raw_index: dict[tuple[str, int], dict[str, Any]] = {}
    raw_by_record_seed: dict[tuple[str, int], dict[str, Any]] = {}
    for seed in SEEDS:
        path = FAILURE_ROOT / f"private/seed{seed}/raw_generations.jsonl"
        report = json.loads(
            (FAILURE_ROOT / f"runs/seed{seed}_report.json").read_text(
                encoding="utf-8"
            )
        )
        if sha256_file(path) != str(report["raw_generations_sha256"]):
            raise ValueError("raw-generation report binding differs")
        for row in read_jsonl(path):
            digest = sha256_text(compact_json(row))
            key = raw_source_key(digest, seed)
            if key in raw_index:
                raise ValueError("duplicate raw-generation evidence hash")
            raw_index[key] = row
            record_key = (str(row["record_id"]), seed)
            if record_key in raw_by_record_seed:
                raise ValueError("duplicate raw-generation record/seed")
            raw_by_record_seed[record_key] = row
    return (
        failure_index,
        raw_index,
        failure_by_record_seed,
        raw_by_record_seed,
    )


def _load_event_indexes() -> tuple[
    dict[str, dict[str, Any]], dict[str, dict[str, Any]]
]:
    frozen = json.loads(FROZEN_MANIFEST_LOCK.read_text(encoding="utf-8"))
    expected = frozen["private_artifact_hashes"]["manifests_by_seed"]
    r2_index: dict[str, dict[str, Any]] = {}
    r3_index: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for arm, target in (("R2", r2_index), ("R3", r3_index)):
            path = DATA_ROOT / f"training_manifest_{arm.lower()}_seed{seed}.jsonl"
            if sha256_file(path) != str(expected[f"seed{seed}"][arm]):
                raise ValueError("materialized manifest binding differs")
            for row in read_jsonl(path):
                event_id = str(row["base_event_id"])
                if event_id in target:
                    raise ValueError("duplicate materialized base event")
                target[event_id] = row
    if set(r2_index) != set(r3_index):
        raise ValueError("R2/R3 event closure differs")
    return r2_index, r3_index


def _validate_score_pair(
    pair: dict[str, Any],
    train_row: dict[str, Any],
    failure_index: dict[str, dict[str, Any]],
    failure_by_record_seed: dict[tuple[str, int], dict[str, Any]],
    r3_index: dict[str, dict[str, Any]],
) -> None:
    if pair["schema_version"] != PAIR_SCHEMA_VERSION:
        raise ValueError("score pair schema differs")
    if _pair_payload_hash(pair, "pair_hash") != str(pair["pair_hash"]):
        raise ValueError("score pair hash differs")
    gold = int(train_row["label_5"])
    rejected = int(pair["rejected"]["score"])
    if (
        int(pair["gold_label"]) != gold
        or int(pair["chosen"]["score"]) != gold
        or rejected == gold
        or pair["chosen"]["rationale"] != pair["rejected"]["rationale"]
        or pair["score_loss_mask"] != "score_value_tokens_only"
        or pair["rationale_loss_mask"] != "off"
        or str(pair["metric_id"]) != str(train_row["metric_id"])
        or str(pair["language"]) != str(train_row["language"])
    ):
        raise ValueError("score pair field-control invariant differs")
    expected_cost = independent_ordinal_cost(gold, rejected)
    expected_offset = independent_odpo_offset(gold, rejected)
    if (
        abs(float(pair["ordinal_cost"]) - expected_cost) > 1e-12
        or abs(float(pair["odpo_offset"]) - expected_offset) > 1e-12
        or int(pair["score_error_distance"]) != abs(rejected - gold)
        or bool(pair["severe_l2h"])
        != (gold <= 2 and rejected >= 4)
        or bool(pair["severe_h2l"])
        != (gold >= 4 and rejected <= 2)
    ):
        raise ValueError("score pair ODPO offset differs")
    rationale = str(pair["chosen"]["rationale"])
    if sha256_text(rationale) != str(pair["rationale_anchor_sha256"]):
        raise ValueError("score pair rationale hash differs")
    anchor_id = pair["rationale_anchor_base_event_id"]
    if anchor_id is None:
        if rationale or pair["rationale_source"] != "none_score_only":
            raise ValueError("empty score anchor contract differs")
    else:
        event = r3_index[str(anchor_id)]
        if (
            str(event["record_id"]) != str(pair["record_id"])
            or not bool(event["rationale_active"])
            or str(event["rationale"]) != rationale
            or str(event["arm_selected_reference_id"])
            != str(pair["rationale_anchor_reference_id"])
        ):
            raise ValueError("score pair anchor source differs")

    source = str(pair["pair_source"])
    evidence_hashes = list(pair["evidence_row_hashes"])
    if source == "actual_controlled":
        if not evidence_hashes:
            raise ValueError("actual pair lacks evidence")
        observed = int(pair["actual_observed_wrong_score"])
        if observed != rejected:
            raise ValueError("actual rejected score differs")
        rows = [failure_index[str(digest)] for digest in evidence_hashes]
        expected_rows = [
            failure_by_record_seed[(str(pair["record_id"]), seed)]
            for seed in SEEDS
            if (
                bool(
                    failure_by_record_seed[
                        (str(pair["record_id"]), seed)
                    ]["parse_success"]
                )
                and int(
                    failure_by_record_seed[
                        (str(pair["record_id"]), seed)
                    ]["generated_score"]
                )
                == rejected
            )
        ]
        expected_hashes = sorted(
            sha256_text(compact_json(row)) for row in expected_rows
        )
        if any(
            str(row["record_id"]) != str(pair["record_id"])
            or int(row["generated_score"]) != rejected
            or not bool(row["parse_success"])
            or int(row["gold_label"]) != gold
            or str(row["metric_id"]) != str(train_row["metric_id"])
            or str(row["language"]) != str(train_row["language"])
            for row in rows
        ) or sorted(str(value) for value in evidence_hashes) != expected_hashes:
            raise ValueError("actual score evidence differs")
        seeds = sorted({int(row["generator_seed"]) for row in rows})
        checkpoints = sorted(
            {str(row["generator_adapter_sha256"]) for row in rows}
        )
        if (
            seeds != list(pair["supporting_generator_seeds"])
            or len(seeds) != int(pair["actual_support_count"])
            or checkpoints != list(pair["generator_checkpoint_hashes"])
            or bool(pair["forced_completion_evidence_any"])
            != any(bool(row["forced_completion"]) for row in rows)
            or bool(pair["forced_completion_evidence_all"])
            != all(bool(row["forced_completion"]) for row in rows)
            or bool(pair["score_leakage_evidence_any"])
            != any(
                score_leakage(
                    str(row["generated_rationale"]),
                    int(row["generated_score"]),
                )
                for row in rows
            )
        ):
            raise ValueError("actual support provenance differs")
    elif source in {"synthetic_backfill", "synthetic_control"}:
        if (
            evidence_hashes
            or pair["actual_observed_wrong_score"] is not None
            or int(pair["actual_support_count"]) != 0
            or list(pair["supporting_generator_seeds"])
            or list(pair["generator_checkpoint_hashes"])
            or bool(pair["forced_completion_evidence_any"])
            or bool(pair["forced_completion_evidence_all"])
            or bool(pair["score_leakage_evidence_any"])
        ):
            raise ValueError("synthetic pair carries actual evidence")
    else:
        raise ValueError("unknown score pair source")


def validate_score_block_rule(pair: dict[str, Any]) -> None:
    block = str(pair["pair_type"])
    gold = int(pair["gold_label"])
    rejected = int(pair["rejected"]["score"])
    source = str(pair["pair_source"])
    if block == "adjacent_score":
        if abs(rejected - gold) != 1 or source != "actual_controlled":
            raise ValueError("adjacent pair rule differs")
        return
    if block == "severe_l2h":
        if not (gold <= 2 and rejected in {4, 5}):
            raise ValueError("severe-L2H pair rule differs")
        if source == "synthetic_backfill" and rejected != 4:
            raise ValueError("severe-L2H backfill score differs")
        return
    if block == "h2l_guard":
        if gold not in {4, 5}:
            raise ValueError("H2L guard gold label differs")
        if source == "actual_controlled" and rejected >= gold:
            raise ValueError("actual H2L is not an underestimate")
        if source == "synthetic_backfill" and rejected != 2:
            raise ValueError("H2L backfill score differs")
        return
    raise ValueError("unknown score block")


def validate_rationale_pair(
    pair: dict[str, Any],
    train_row: dict[str, Any],
    r2_index: dict[str, dict[str, Any]],
    r3_index: dict[str, dict[str, Any]],
) -> None:
    if (
        pair["schema_version"] != PAIR_SCHEMA_VERSION
        or _pair_payload_hash(pair, "pair_hash") != str(pair["pair_hash"])
        or pair["pair_type"] != "rationale_alignment"
        or pair["pair_source"] != "rationale_control"
        or pair["score_loss_mask"] != "off"
        or pair["rationale_loss_mask"]
        != "rationale_content_tokens_only"
    ):
        raise ValueError("rationale pair envelope differs")
    record_id = str(pair["record_id"])
    gold = int(train_row["label_5"])
    chosen_score = int(pair["chosen"]["score"])
    rejected_score = int(pair["rejected"]["score"])
    chosen = str(pair["chosen"]["rationale"])
    rejected = str(pair["rejected"]["rationale"])
    if (
        int(pair["gold_label"]) != gold
        or chosen_score != gold
        or rejected_score != gold
        or str(pair["metric_id"]) != str(train_row["metric_id"])
        or str(pair["language"]) != str(train_row["language"])
        or not chosen
        or not rejected
        or normalize_text(chosen) == normalize_text(rejected)
        or any_explicit_score_leakage(chosen)
        or any_explicit_score_leakage(rejected)
        or sha256_text(chosen) != str(pair["chosen_rationale_sha256"])
        or sha256_text(rejected)
        != str(pair["rejected_rationale_sha256"])
        or float(pair["ordinal_cost"]) != 0.0
        or float(pair["odpo_offset"]) != 0.0
    ):
        raise ValueError("rationale pair train/field contract differs")
    event_id = str(pair["base_event_id"])
    if event_id not in r2_index or event_id not in r3_index:
        raise ValueError("rationale base event is absent")
    r2 = r2_index[event_id]
    r3 = r3_index[event_id]
    if (
        str(r3["record_id"]) != record_id
        or str(r2["record_id"]) != record_id
        or int(r3["score_target"]) != gold
        or int(r2["score_target"]) != gold
        or not bool(r3["rationale_active"])
        or not bool(r2["rationale_active"])
        or str(r3["rationale"]) != chosen
        or str(r2["rationale"]) != rejected
        or str(r3["arm_selected_reference_id"])
        != str(pair["r3_reference_id"])
        or str(r2["arm_selected_reference_id"])
        != str(pair["r2_donor_reference_id"])
        or str(r2["arm_rationale_source_event_id"])
        != str(pair["r2_donor_event_id"])
    ):
        raise ValueError("rationale pair frozen-event source differs")


def independent_anchor(
    record_id: str,
    r3_index: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in r3_index.values()
        if (
            str(row["record_id"]) == record_id
            and bool(row["rationale_active"])
        )
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            stable_hash(
                "exp54-pref-anchor-v1",
                record_id,
                row["base_event_id"],
            ),
            str(row["base_event_id"]),
        ),
    )


def expected_actual_candidate_source(
    *,
    record_id: str,
    failure_by_record_seed: dict[tuple[str, int], dict[str, Any]],
    raw_by_record_seed: dict[tuple[str, int], dict[str, Any]],
    r3_index: dict[str, dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any] | None:
    anchor = independent_anchor(record_id, r3_index)
    if anchor is None:
        return None
    human = str(anchor["rationale"])
    human_tokens = len(list(anchor["rationale_token_ids"]))
    eligible: list[dict[str, Any]] = []
    for seed in SEEDS:
        failure = failure_by_record_seed[(record_id, seed)]
        raw = raw_by_record_seed[(record_id, seed)]
        model = str(failure.get("generated_rationale") or "")
        if (
            not bool(failure["parse_success"])
            or int(failure["generated_score"]) != int(failure["gold_label"])
            or bool(failure["forced_completion"])
            or bool(raw["forced_completion"])
            or str(raw["finish_reason"]) == "length"
            or int(raw["backend_generated_tokens"]) >= 256
            or not model.strip()
            or any_explicit_score_leakage(model)
            or normalize_text(human) == normalize_text(model)
        ):
            continue
        model_tokens = len(
            tokenizer.encode(model, add_special_tokens=False)
        )
        eligible.append(
            {
                "failure": failure,
                "raw": raw,
                "anchor": anchor,
                "human": human,
                "human_token_count": human_tokens,
                "model": model,
                "model_token_count": model_tokens,
            }
        )
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda value: (
            abs(
                int(value["model_token_count"])
                - int(value["human_token_count"])
            ),
            stable_hash(
                "actual-rationale-candidate",
                record_id,
                value["failure"]["generator_seed"],
                sha256_text(str(value["failure"]["generated_rationale"])),
            ),
        ),
    )


def validate_actual_candidate(
    *,
    candidate: dict[str, Any],
    train_row: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    if (
        _pair_payload_hash(candidate, "candidate_hash")
        != str(candidate["candidate_hash"])
        or candidate["schema_version"]
        != "exp54-actual-rationale-candidate-v1"
        or candidate["preference_label_status"]
        != "PENDING_TWO_FAMILY_BLIND_REVIEW"
        or candidate["preference_training_allowed"]
        or candidate["forced_completion"]
        or candidate["score_leakage"]
    ):
        raise ValueError("actual-rationale candidate envelope differs")
    failure = expected["failure"]
    raw = expected["raw"]
    anchor = expected["anchor"]
    record_id = str(train_row["record_id"])
    gold = int(train_row["label_5"])
    human = str(expected["human"])
    model = str(expected["model"])
    if (
        str(candidate["record_id"]) != record_id
        or int(candidate["gold_label"]) != gold
        or str(candidate["metric_id"]) != str(train_row["metric_id"])
        or str(candidate["language"]) != str(train_row["language"])
        or int(candidate["human_output"]["score"]) != gold
        or int(candidate["model_output"]["score"]) != gold
        or str(candidate["human_output"]["rationale"]) != human
        or str(candidate["model_output"]["rationale"]) != model
        or str(candidate["human_rationale_sha256"]) != sha256_text(human)
        or str(candidate["model_rationale_sha256"]) != sha256_text(model)
        or int(candidate["human_rationale_token_count"])
        != int(expected["human_token_count"])
        or int(candidate["model_rationale_token_count"])
        != int(expected["model_token_count"])
        or int(candidate["source_generator_seed"])
        != int(failure["generator_seed"])
        or str(candidate["source_checkpoint_sha256"])
        != str(failure["generator_adapter_sha256"])
        or str(candidate["source_failure_row_sha256"])
        != sha256_text(compact_json(failure))
        or str(candidate["source_raw_generation_sha256"])
        != sha256_text(compact_json(raw))
        or str(anchor["record_id"]) != record_id
    ):
        raise ValueError("actual-rationale candidate source binding differs")


def audit(*, write: bool) -> dict[str, Any]:
    lock_path = OUTPUT_ROOT / "candidate_lock.json"
    report_path = OUTPUT_ROOT / "candidate_report.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    builder_path = (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/build_sorc_dpo_pairs.py"
    )
    if (
        lock["status"] != "PAIR_CANDIDATES_NOT_FROZEN_TRAINING_NOT_ALLOWED"
        or lock["preference_training_allowed"]
        or lock["dev_accessed"]
        or lock["test_accessed"]
        or sha256_file(report_path) != str(lock["candidate_report_sha256"])
        or sha256_file(PROTOCOL_PATH) != str(lock["protocol_sha256"])
        or sha256_file(FAILURE_PATH) != str(lock["failure_bank_sha256"])
        or sha256_file(builder_path)
        != str(lock["builder_source_sha256"])
        or sha256_file(Path(__file__))
        != str(lock["independent_auditor_source_sha256"])
        or report.get("builder_source_sha256")
        != lock["builder_source_sha256"]
        or report.get("independent_auditor_source_sha256")
        != lock["independent_auditor_source_sha256"]
    ):
        raise ValueError("candidate public lock boundary differs")

    paths = {
        name: OUTPUT_ROOT / "private" / f"{name}.jsonl"
        for name in (
            "score_pairs_hybrid",
            "score_pairs_synthetic_seed42",
            "rationale_pairs_r3_r2",
            "actual_rationale_candidates",
        )
    }
    for name, path in paths.items():
        if sha256_file(path) != str(lock["private_output_hashes"][name]):
            raise ValueError(f"{name}: private output hash differs")
    hybrid = read_jsonl(paths["score_pairs_hybrid"])
    synthetic = read_jsonl(paths["score_pairs_synthetic_seed42"])
    rationale = read_jsonl(paths["rationale_pairs_r3_r2"])
    actual_candidates = read_jsonl(paths["actual_rationale_candidates"])
    train_rows = load_train_rows()
    train_by_id = {str(row["record_id"]): row for row in train_rows}
    (
        failure_index,
        raw_index,
        failure_by_record_seed,
        raw_by_record_seed,
    ) = _index_failure_sources()
    r2_index, r3_index = _load_event_indexes()
    training_config = json.loads(
        TRAINING_CONFIG_PATH.read_text(encoding="utf-8")
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(training_config["model"]["local_path"]),
        local_files_only=True,
        trust_remote_code=False,
    )

    block_ids: dict[str, set[str]] = defaultdict(set)
    for pair in hybrid:
        record_id = str(pair["record_id"])
        if record_id not in train_by_id:
            raise ValueError("hybrid pair is outside train")
        _validate_score_pair(
            pair,
            train_by_id[record_id],
            failure_index,
            failure_by_record_seed,
            r3_index,
        )
        block = str(pair["pair_type"])
        if block not in BLOCKS or record_id in block_ids[block]:
            raise ValueError("hybrid block record uniqueness differs")
        block_ids[block].add(record_id)
        gold = int(pair["gold_label"])
        rejected = int(pair["rejected"]["score"])
        validate_score_block_rule(pair)

    low = [row for row in hybrid if row["pair_type"] == "severe_l2h"]
    h2l = [row for row in hybrid if row["pair_type"] == "h2l_guard"]
    if (
        len(low) != 76
        or Counter(row["pair_source"] for row in low)
        != {"actual_controlled": 53, "synthetic_backfill": 23}
        or len(h2l) != 76
        or Counter(int(row["gold_label"]) for row in h2l)
        != {4: 52, 5: 24}
        or len({str(row["record_id"]) for row in h2l}) != 76
        or len({str(row["mirrored_low_record_id"]) for row in h2l}) != 76
    ):
        raise ValueError("risk block aggregate contract differs")
    for pair in h2l:
        low_row = train_by_id[str(pair["mirrored_low_record_id"])]
        if int(pair["gold_label"]) != 6 - int(low_row["label_5"]):
            raise ValueError("H2L mirror label differs")

    hybrid_by_key = {
        (str(row["pair_type"]), str(row["record_id"])): row
        for row in hybrid
    }
    if len(synthetic) != len(hybrid):
        raise ValueError("matched synthetic pair count differs")
    for pair in synthetic:
        record_id = str(pair["record_id"])
        _validate_score_pair(
            pair,
            train_by_id[record_id],
            failure_index,
            failure_by_record_seed,
            r3_index,
        )
        key = (str(pair["pair_type"]), record_id)
        source = hybrid_by_key.get(key)
        if source is None:
            raise ValueError("synthetic pair lacks hybrid match")
        for field in (
            "gold_label",
            "metric_id",
            "language",
            "chosen",
            "rationale_anchor_sha256",
            "mirrored_low_record_id",
        ):
            if pair[field] != source[field]:
                raise ValueError("synthetic/hybrid matched field differs")
        if pair["pair_source"] != "synthetic_control":
            raise ValueError("synthetic diagnostic source differs")
        expected_rejected = independent_synthetic_score(
            str(pair["pair_type"]),
            record_id,
            int(pair["gold_label"]),
        )
        if int(pair["rejected"]["score"]) != expected_rejected:
            raise ValueError("synthetic diagnostic rejected score differs")

    rationale_ids: set[str] = set()
    for pair in rationale:
        record_id = str(pair["record_id"])
        if record_id not in train_by_id:
            raise ValueError("rationale pair is outside train")
        validate_rationale_pair(
            pair,
            train_by_id[record_id],
            r2_index,
            r3_index,
        )
        if record_id in rationale_ids:
            raise ValueError("duplicate rationale record")
        rationale_ids.add(record_id)

    expected_actual = {
        record_id: expected
        for record_id in train_by_id
        if (
            expected := expected_actual_candidate_source(
                record_id=record_id,
                failure_by_record_seed=failure_by_record_seed,
                raw_by_record_seed=raw_by_record_seed,
                r3_index=r3_index,
                tokenizer=tokenizer,
            )
        )
        is not None
    }
    candidate_ids: set[str] = set()
    for candidate in actual_candidates:
        record_id = str(candidate["record_id"])
        if record_id in candidate_ids:
            raise ValueError("duplicate actual-rationale record")
        if record_id not in train_by_id or record_id not in expected_actual:
            raise ValueError("unexpected actual-rationale candidate record")
        candidate_ids.add(record_id)
        validate_actual_candidate(
            candidate=candidate,
            train_row=train_by_id[record_id],
            expected=expected_actual[record_id],
        )
    if candidate_ids != set(expected_actual):
        raise ValueError("actual-rationale candidate record closure differs")

    aggregate = {
        "hybrid_score_pairs": len(hybrid),
        "hybrid_unique_records": len(
            {str(row["record_id"]) for row in hybrid}
        ),
        "hybrid_pair_types": dict(
            sorted(Counter(row["pair_type"] for row in hybrid).items())
        ),
        "hybrid_pair_sources": dict(
            sorted(Counter(row["pair_source"] for row in hybrid).items())
        ),
        "synthetic_score_pairs": len(synthetic),
        "rationale_pairs": len(rationale),
        "rationale_unique_records": len(rationale_ids),
        "actual_rationale_candidates": len(actual_candidates),
        "actual_rationale_minimum_128_met": len(actual_candidates) >= 128,
    }
    for key, value in aggregate.items():
        if report["aggregate"].get(key) != value:
            raise ValueError(f"candidate report aggregate differs at {key}")
    result = {
        "schema_version": "exp54-sorc-dpo-pair-audit-v1",
        "status": "SORC_DPO_PAIR_CANDIDATE_AUDIT_PASS",
        "aggregate": aggregate,
        "candidate_lock_sha256": sha256_file(lock_path),
        "candidate_report_sha256": sha256_file(report_path),
        "builder_source_sha256": sha256_file(
            builder_path
        ),
        "auditor_source_sha256": sha256_file(Path(__file__)),
        "pair_freeze_allowed": False,
        "rationale_blind_qualification_completed": False,
        "actual_rationale_preference_labels_created": False,
        "preference_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    if write:
        path = OUTPUT_ROOT / "audit_report.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            audit(write=args.write),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
