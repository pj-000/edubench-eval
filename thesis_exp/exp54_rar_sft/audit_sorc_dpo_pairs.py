"""Independently audit private SORC-DPO candidate pair manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import (
    SEEDS,
    load_train_rows,
    read_jsonl,
    score_leakage,
    sha256_file,
)
from thesis_exp.exp54_rar_sft.build_sorc_dpo_pairs import (
    BLOCKS,
    DATA_ROOT,
    FAILURE_PATH,
    FAILURE_ROOT,
    FROZEN_MANIFEST_LOCK,
    OUTPUT_ROOT,
    PAIR_SCHEMA_VERSION,
    PROTOCOL_PATH,
    compact_json,
    odpo_offset,
    sha256_text,
)


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
]:
    failure_index: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(FAILURE_PATH):
        digest = sha256_text(compact_json(row))
        if digest in failure_index:
            raise ValueError("duplicate failure evidence hash")
        failure_index[digest] = row
    raw_index: dict[tuple[str, int], dict[str, Any]] = {}
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
    return failure_index, raw_index


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
    if abs(float(pair["odpo_offset"]) - odpo_offset(gold, rejected)) > 1e-12:
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
        if any(
            str(row["record_id"]) != str(pair["record_id"])
            or int(row["generated_score"]) != rejected
            or not bool(row["parse_success"])
            for row in rows
        ):
            raise ValueError("actual score evidence differs")
        seeds = sorted({int(row["generator_seed"]) for row in rows})
        if (
            seeds != list(pair["supporting_generator_seeds"])
            or len(seeds) != int(pair["actual_support_count"])
        ):
            raise ValueError("actual support provenance differs")
    elif source in {"synthetic_backfill", "synthetic_control"}:
        if (
            evidence_hashes
            or pair["actual_observed_wrong_score"] is not None
            or int(pair["actual_support_count"]) != 0
        ):
            raise ValueError("synthetic pair carries actual evidence")
    else:
        raise ValueError("unknown score pair source")


def audit(*, write: bool) -> dict[str, Any]:
    lock_path = OUTPUT_ROOT / "candidate_lock.json"
    report_path = OUTPUT_ROOT / "candidate_report.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        lock["status"] != "PAIR_CANDIDATES_NOT_FROZEN_TRAINING_NOT_ALLOWED"
        or lock["preference_training_allowed"]
        or lock["dev_accessed"]
        or lock["test_accessed"]
        or sha256_file(report_path) != str(lock["candidate_report_sha256"])
        or sha256_file(PROTOCOL_PATH) != str(lock["protocol_sha256"])
        or sha256_file(FAILURE_PATH) != str(lock["failure_bank_sha256"])
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
    failure_index, raw_index = _index_failure_sources()
    r2_index, r3_index = _load_event_indexes()

    block_ids: dict[str, set[str]] = defaultdict(set)
    for pair in hybrid:
        record_id = str(pair["record_id"])
        if record_id not in train_by_id:
            raise ValueError("hybrid pair is outside train")
        _validate_score_pair(
            pair, train_by_id[record_id], failure_index, r3_index
        )
        block = str(pair["pair_type"])
        if block not in BLOCKS or record_id in block_ids[block]:
            raise ValueError("hybrid block record uniqueness differs")
        block_ids[block].add(record_id)
        gold = int(pair["gold_label"])
        rejected = int(pair["rejected"]["score"])
        if block == "adjacent_score" and (
            abs(rejected - gold) != 1
            or pair["pair_source"] != "actual_controlled"
        ):
            raise ValueError("adjacent pair rule differs")
        if block == "severe_l2h" and not (
            gold <= 2 and rejected >= 4
        ):
            raise ValueError("severe-L2H pair rule differs")
        if block == "h2l_guard" and gold not in {4, 5}:
            raise ValueError("H2L guard gold label differs")

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
            pair, train_by_id[record_id], failure_index, r3_index
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

    rationale_ids: set[str] = set()
    for pair in rationale:
        if (
            pair["schema_version"] != PAIR_SCHEMA_VERSION
            or _pair_payload_hash(pair, "pair_hash") != str(pair["pair_hash"])
            or pair["pair_type"] != "rationale_alignment"
            or pair["pair_source"] != "rationale_control"
            or pair["chosen"]["score"] != pair["rejected"]["score"]
            or pair["chosen"]["rationale"] == pair["rejected"]["rationale"]
            or pair["score_loss_mask"] != "off"
            or pair["rationale_loss_mask"]
            != "rationale_content_tokens_only"
        ):
            raise ValueError("rationale pair field-control invariant differs")
        record_id = str(pair["record_id"])
        if record_id in rationale_ids:
            raise ValueError("duplicate rationale record")
        rationale_ids.add(record_id)
        event_id = str(pair["base_event_id"])
        r2 = r2_index[event_id]
        r3 = r3_index[event_id]
        if (
            str(r3["record_id"]) != record_id
            or str(r3["rationale"]) != str(pair["chosen"]["rationale"])
            or str(r2["rationale"]) != str(pair["rejected"]["rationale"])
            or str(r3["arm_selected_reference_id"])
            != str(pair["r3_reference_id"])
            or str(r2["arm_selected_reference_id"])
            != str(pair["r2_donor_reference_id"])
        ):
            raise ValueError("rationale pair frozen-event source differs")

    candidate_ids: set[str] = set()
    for candidate in actual_candidates:
        if (
            _pair_payload_hash(candidate, "candidate_hash")
            != str(candidate["candidate_hash"])
            or candidate["preference_label_status"]
            != "PENDING_TWO_FAMILY_BLIND_REVIEW"
            or candidate["preference_training_allowed"]
            or candidate["human_output"]["score"]
            != candidate["model_output"]["score"]
            or not str(candidate["human_output"]["rationale"]).strip()
            or not str(candidate["model_output"]["rationale"]).strip()
        ):
            raise ValueError("actual-rationale candidate contract differs")
        record_id = str(candidate["record_id"])
        if record_id in candidate_ids:
            raise ValueError("duplicate actual-rationale record")
        candidate_ids.add(record_id)
        failure = failure_index[str(candidate["source_failure_row_sha256"])]
        raw = raw_index[
            raw_source_key(
                str(candidate["source_raw_generation_sha256"]),
                int(candidate["source_generator_seed"]),
            )
        ]
        score = int(failure["generated_score"])
        if (
            str(failure["record_id"]) != record_id
            or not bool(failure["parse_success"])
            or score != int(failure["gold_label"])
            or bool(failure["forced_completion"])
            or str(raw["finish_reason"]) == "length"
            or int(raw["backend_generated_tokens"]) >= 256
            or score_leakage(str(failure["generated_rationale"]), score)
        ):
            raise ValueError("actual-rationale source eligibility differs")

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
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/build_sorc_dpo_pairs.py"
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
