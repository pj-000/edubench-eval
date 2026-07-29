"""Build train-only SORC-DPO preference-pair candidates.

This stage materializes private pair text and public aggregate hashes. It does
not label actual model rationales, train a model, or read dev/test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import (
    SEEDS,
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
FAILURE_REPORT = FAILURE_ROOT / "aggregate_report.json"
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
BLOCKS = ("adjacent_score", "severe_l2h", "h2l_guard")
LOW_LABELS = {1, 2}
HIGH_LABELS = {4, 5}
PAIR_SCHEMA_VERSION = "exp54-sorc-dpo-pair-v1"


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def ordinal_cost(gold: int, rejected: int) -> float:
    if gold not in range(1, 6) or rejected not in range(1, 6):
        raise ValueError("score outside 1-5")
    severe_l2h = int(gold <= 2 and rejected >= 4)
    return abs(rejected - gold) / 4.0 + severe_l2h


def odpo_offset(gold: int, rejected: int) -> float:
    return ordinal_cost(gold, rejected) / 2.0


@dataclass(frozen=True)
class ScoreEvidence:
    score: int
    rows: tuple[dict[str, Any], ...]

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(sorted({int(row["generator_seed"]) for row in self.rows}))

    @property
    def support(self) -> int:
        return len(self.seeds)

    @property
    def forced_any(self) -> bool:
        return any(bool(row["forced_completion"]) for row in self.rows)

    @property
    def forced_all(self) -> bool:
        return all(bool(row["forced_completion"]) for row in self.rows)

    @property
    def leakage_any(self) -> bool:
        return any(
            score_leakage(
                str(row["generated_rationale"]),
                int(row["generated_score"]),
            )
            for row in self.rows
        )

    @property
    def checkpoint_hashes(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    str(row["generator_adapter_sha256"])
                    for row in self.rows
                }
            )
        )

    @property
    def evidence_hashes(self) -> tuple[str, ...]:
        return tuple(sorted(sha256_text(compact_json(row)) for row in self.rows))


def group_failure_evidence(
    train_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> dict[str, dict[int, ScoreEvidence]]:
    if len(failure_rows) != len(train_rows) * len(SEEDS):
        raise ValueError("failure-bank row count differs")
    train_by_id = {str(row["record_id"]): row for row in train_rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in failure_rows:
        record_id = str(row["record_id"])
        if record_id not in train_by_id:
            raise ValueError("failure-bank record is outside train")
        if (
            int(row["generator_seed"]) not in SEEDS
            or row["generator_arm"] != "R3"
            or int(row["generator_epoch"]) != 3
            or row["generation_mode"] != "greedy"
            or row["rollout_seed"] is not None
        ):
            raise ValueError("failure-bank provenance differs")
        source = train_by_id[record_id]
        if (
            int(row["gold_label"]) != int(source["label_5"])
            or str(row["metric_id"]) != str(source["metric_id"])
            or str(row["language"]) != str(source["language"])
        ):
            raise ValueError("failure-bank train metadata differs")
        grouped[record_id].append(row)

    output: dict[str, dict[int, ScoreEvidence]] = {}
    for record_id in train_by_id:
        rows = grouped[record_id]
        if len(rows) != len(SEEDS):
            raise ValueError(f"{record_id}: expected three policy observations")
        if {int(row["generator_seed"]) for row in rows} != set(SEEDS):
            raise ValueError(f"{record_id}: policy seed closure differs")
        by_score: dict[int, list[dict[str, Any]]] = defaultdict(list)
        gold = int(train_by_id[record_id]["label_5"])
        for row in rows:
            if not bool(row["parse_success"]):
                continue
            score = int(row["generated_score"])
            if score != gold:
                by_score[score].append(row)
        output[record_id] = {
            score: ScoreEvidence(
                score=score,
                rows=tuple(
                    sorted(
                        evidence_rows,
                        key=lambda item: int(item["generator_seed"]),
                    )
                ),
            )
            for score, evidence_rows in by_score.items()
        }
    return output


def choose_adjacent(
    record_id: str,
    gold: int,
    evidence: dict[int, ScoreEvidence],
) -> ScoreEvidence | None:
    candidates = [
        value for score, value in evidence.items() if abs(score - gold) == 1
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda value: (
            -value.support,
            not (value.score > gold),
            stable_hash("adjacent", record_id, value.score),
        ),
    )


def choose_severe_l2h(
    record_id: str,
    gold: int,
    evidence: dict[int, ScoreEvidence],
) -> ScoreEvidence | None:
    candidates = [
        value for score, value in evidence.items() if score in {4, 5}
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda value: (
            -value.support,
            abs(value.score - gold),
            value.score,
            stable_hash("l2h", record_id, value.score),
        ),
    )


def choose_underestimate(
    record_id: str,
    gold: int,
    evidence: dict[int, ScoreEvidence],
) -> ScoreEvidence | None:
    candidates = [value for score, value in evidence.items() if score < gold]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda value: (
            not (gold >= 4 and value.score <= 2),
            -abs(value.score - gold),
            -value.support,
            stable_hash("h2l", record_id, value.score),
        ),
    )


def _load_materialized_events(
    frozen_lock: dict[str, Any],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, tuple[dict[str, Any], dict[str, Any]]],
    dict[str, str],
]:
    events_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paired_by_event: dict[
        str, tuple[dict[str, Any], dict[str, Any]]
    ] = {}
    hashes: dict[str, str] = {}
    expected = frozen_lock["private_artifact_hashes"]["manifests_by_seed"]
    for seed in SEEDS:
        by_arm: dict[str, dict[str, dict[str, Any]]] = {}
        for arm in ("R2", "R3"):
            path = DATA_ROOT / f"training_manifest_{arm.lower()}_seed{seed}.jsonl"
            actual_hash = sha256_file(path)
            expected_hash = str(expected[f"seed{seed}"][arm])
            if actual_hash != expected_hash:
                raise ValueError(f"{path}: frozen manifest hash differs")
            rows = read_jsonl(path)
            if len(rows) != 7962:
                raise ValueError(f"{path}: expected 7,962 events")
            index: dict[str, dict[str, Any]] = {}
            for row in rows:
                event_id = str(row["base_event_id"])
                if event_id in index:
                    raise ValueError(f"{path}: duplicate base event")
                if (
                    str(row["arm"]) != arm
                    or int(row["seed"]) != seed
                    or bool(row["truncated"])
                    or not bool(row["score_loss_active"])
                ):
                    raise ValueError(f"{event_id}: materialized event differs")
                if sha256_text(str(row["rationale"])) != str(
                    row["rationale_bytes_sha256"]
                ):
                    raise ValueError(f"{event_id}: rationale hash differs")
                index[event_id] = row
            by_arm[arm] = index
            hashes[f"{arm}_seed{seed}"] = actual_hash
        if set(by_arm["R2"]) != set(by_arm["R3"]):
            raise ValueError(f"seed{seed}: R2/R3 event closure differs")
        for event_id, r3 in by_arm["R3"].items():
            r2 = by_arm["R2"][event_id]
            if (
                str(r2["record_id"]) != str(r3["record_id"])
                or int(r2["score_target"]) != int(r3["score_target"])
                or bool(r2["rationale_active"])
                != bool(r3["rationale_active"])
            ):
                raise ValueError(f"{event_id}: R2/R3 semantics differ")
            paired_by_event[event_id] = (r2, r3)
            if bool(r3["rationale_active"]):
                events_by_record[str(r3["record_id"])].append(r3)
    return events_by_record, paired_by_event, hashes


def choose_anchor(
    record_id: str,
    events_by_record: dict[str, list[dict[str, Any]]],
) -> tuple[str, dict[str, Any] | None]:
    candidates = events_by_record.get(record_id, [])
    if not candidates:
        return "", None
    selected = min(
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
    return str(selected["rationale"]), selected


def evidence_metadata(value: ScoreEvidence | None) -> dict[str, Any]:
    if value is None:
        return {
            "actual_observed_wrong_score": None,
            "actual_support_count": 0,
            "supporting_generator_seeds": [],
            "generator_checkpoint_hashes": [],
            "evidence_row_hashes": [],
            "forced_completion_evidence_any": False,
            "forced_completion_evidence_all": False,
            "score_leakage_evidence_any": False,
        }
    return {
        "actual_observed_wrong_score": value.score,
        "actual_support_count": value.support,
        "supporting_generator_seeds": list(value.seeds),
        "generator_checkpoint_hashes": list(value.checkpoint_hashes),
        "evidence_row_hashes": list(value.evidence_hashes),
        "forced_completion_evidence_any": value.forced_any,
        "forced_completion_evidence_all": value.forced_all,
        "score_leakage_evidence_any": value.leakage_any,
    }


def make_score_pair(
    *,
    train_row: dict[str, Any],
    rationale: str,
    anchor_event: dict[str, Any] | None,
    rejected_score: int,
    pair_type: str,
    pair_source: str,
    evidence: ScoreEvidence | None,
    mirrored_low_record_id: str | None = None,
) -> dict[str, Any]:
    gold = int(train_row["label_5"])
    if rejected_score == gold:
        raise ValueError("score pair cannot use the gold score as rejected")
    chosen = {"score": gold, "rationale": rationale}
    rejected = {"score": rejected_score, "rationale": rationale}
    pair = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "record_id": str(train_row["record_id"]),
        "gold_label": gold,
        "metric_id": str(train_row["metric_id"]),
        "language": str(train_row["language"]),
        "chosen": chosen,
        "rejected": rejected,
        "pair_type": pair_type,
        "pair_source": pair_source,
        "mirrored_low_record_id": mirrored_low_record_id,
        "rationale_source": (
            "r3_aligned_human"
            if anchor_event is not None
            else "none_score_only"
        ),
        "rationale_anchor_base_event_id": (
            str(anchor_event["base_event_id"])
            if anchor_event is not None
            else None
        ),
        "rationale_anchor_reference_id": (
            str(anchor_event["arm_selected_reference_id"])
            if anchor_event is not None
            else None
        ),
        "rationale_anchor_sha256": sha256_text(rationale),
        "score_loss_mask": "score_value_tokens_only",
        "rationale_loss_mask": "off",
        "score_error_distance": abs(rejected_score - gold),
        "severe_l2h": gold <= 2 and rejected_score >= 4,
        "severe_h2l": gold >= 4 and rejected_score <= 2,
        "ordinal_cost": ordinal_cost(gold, rejected_score),
        "odpo_offset": odpo_offset(gold, rejected_score),
        **evidence_metadata(evidence),
    }
    pair["pair_hash"] = sha256_text(compact_json(pair))
    return pair


def build_hybrid_score_pairs(
    train_rows: list[dict[str, Any]],
    grouped: dict[str, dict[int, ScoreEvidence]],
    events_by_record: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    by_id = {str(row["record_id"]): row for row in train_rows}
    anchors = {
        record_id: choose_anchor(record_id, events_by_record)
        for record_id in by_id
    }

    for row in train_rows:
        record_id = str(row["record_id"])
        gold = int(row["label_5"])
        selected = choose_adjacent(record_id, gold, grouped[record_id])
        if selected is not None:
            rationale, anchor = anchors[record_id]
            pairs.append(
                make_score_pair(
                    train_row=row,
                    rationale=rationale,
                    anchor_event=anchor,
                    rejected_score=selected.score,
                    pair_type="adjacent_score",
                    pair_source="actual_controlled",
                    evidence=selected,
                )
            )

    low_rows = [row for row in train_rows if int(row["label_5"]) in LOW_LABELS]
    if len(low_rows) != 76:
        raise ValueError("low-label train inventory differs")
    for row in low_rows:
        record_id = str(row["record_id"])
        gold = int(row["label_5"])
        selected = choose_severe_l2h(record_id, gold, grouped[record_id])
        rationale, anchor = anchors[record_id]
        pairs.append(
            make_score_pair(
                train_row=row,
                rationale=rationale,
                anchor_event=anchor,
                rejected_score=selected.score if selected is not None else 4,
                pair_type="severe_l2h",
                pair_source=(
                    "actual_controlled"
                    if selected is not None
                    else "synthetic_backfill"
                ),
                evidence=selected,
            )
        )

    unused_high = {
        str(row["record_id"]): row
        for row in train_rows
        if int(row["label_5"]) in HIGH_LABELS
    }
    recipients = sorted(
        low_rows,
        key=lambda row: stable_hash("h2l-recipient", row["record_id"]),
    )
    for low_row in recipients:
        mirror_label = 6 - int(low_row["label_5"])
        candidates = [
            row
            for row in unused_high.values()
            if int(row["label_5"]) == mirror_label
        ]
        if not candidates:
            raise ValueError("H2L mirror candidate pool exhausted")

        def rank(high_row: dict[str, Any]) -> tuple[Any, ...]:
            high_id = str(high_row["record_id"])
            actual = choose_underestimate(
                high_id,
                int(high_row["label_5"]),
                grouped[high_id],
            )
            return (
                actual is None,
                str(high_row["metric_id"]) != str(low_row["metric_id"]),
                str(high_row["language"]) != str(low_row["language"]),
                stable_hash(
                    "h2l-match",
                    low_row["record_id"],
                    high_id,
                ),
            )

        selected_row = min(candidates, key=rank)
        selected_id = str(selected_row["record_id"])
        unused_high.pop(selected_id)
        evidence = choose_underestimate(
            selected_id,
            int(selected_row["label_5"]),
            grouped[selected_id],
        )
        rationale, anchor = anchors[selected_id]
        pairs.append(
            make_score_pair(
                train_row=selected_row,
                rationale=rationale,
                anchor_event=anchor,
                rejected_score=evidence.score if evidence is not None else 2,
                pair_type="h2l_guard",
                pair_source=(
                    "actual_controlled"
                    if evidence is not None
                    else "synthetic_backfill"
                ),
                evidence=evidence,
                mirrored_low_record_id=str(low_row["record_id"]),
            )
        )

    for block in BLOCKS:
        block_ids = [
            str(pair["record_id"])
            for pair in pairs
            if pair["pair_type"] == block
        ]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError(f"{block}: duplicate record")
    low_block = [pair for pair in pairs if pair["pair_type"] == "severe_l2h"]
    if Counter(pair["pair_source"] for pair in low_block) != {
        "actual_controlled": 53,
        "synthetic_backfill": 23,
    }:
        raise ValueError("frozen severe-L2H actual/backfill counts differ")
    h2l = [pair for pair in pairs if pair["pair_type"] == "h2l_guard"]
    if len(h2l) != 76 or Counter(pair["gold_label"] for pair in h2l) != {
        4: 52,
        5: 24,
    }:
        raise ValueError("mirrored H2L guard counts differ")
    return sorted(pairs, key=lambda pair: (pair["pair_type"], pair["record_id"]))


def _synthetic_adjacent_score(record_id: str, gold: int) -> int:
    if gold == 1:
        return 2
    if gold == 5:
        return 4
    direction = 1 if int(stable_hash("synthetic-adjacent", record_id), 16) % 2 else -1
    return gold + direction


def build_matched_synthetic_pairs(
    hybrid_pairs: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    events_by_record: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_id = {str(row["record_id"]): row for row in train_rows}
    output = []
    for hybrid in hybrid_pairs:
        record_id = str(hybrid["record_id"])
        row = by_id[record_id]
        gold = int(row["label_5"])
        pair_type = str(hybrid["pair_type"])
        if pair_type == "adjacent_score":
            rejected = _synthetic_adjacent_score(record_id, gold)
        elif pair_type == "severe_l2h":
            rejected = 4
        elif pair_type == "h2l_guard":
            rejected = 2
        else:
            raise ValueError("unknown score pair block")
        rationale, anchor = choose_anchor(record_id, events_by_record)
        output.append(
            make_score_pair(
                train_row=row,
                rationale=rationale,
                anchor_event=anchor,
                rejected_score=rejected,
                pair_type=pair_type,
                pair_source="synthetic_control",
                evidence=None,
                mirrored_low_record_id=hybrid["mirrored_low_record_id"],
            )
        )
    return sorted(output, key=lambda pair: (pair["pair_type"], pair["record_id"]))


def build_rationale_pairs(
    train_rows: list[dict[str, Any]],
    paired_by_event: dict[
        str, tuple[dict[str, Any], dict[str, Any]]
    ],
) -> list[dict[str, Any]]:
    train_by_id = {str(row["record_id"]): row for row in train_rows}
    candidates: dict[
        str, list[tuple[dict[str, Any], dict[str, Any]]]
    ] = defaultdict(list)
    for r2, r3 in paired_by_event.values():
        if not bool(r3["rationale_active"]):
            continue
        chosen = str(r3["rationale"])
        rejected = str(r2["rationale"])
        if (
            not chosen
            or not rejected
            or normalize_text(chosen) == normalize_text(rejected)
            or score_leakage(chosen, int(r3["score_target"]))
            or score_leakage(rejected, int(r3["score_target"]))
        ):
            continue
        candidates[str(r3["record_id"])].append((r2, r3))

    output = []
    for record_id, rows in candidates.items():
        r2, r3 = min(
            rows,
            key=lambda value: (
                stable_hash(
                    "exp54-rationale-pair-v1",
                    record_id,
                    value[1]["base_event_id"],
                ),
                str(value[1]["base_event_id"]),
            ),
        )
        train_row = train_by_id[record_id]
        gold = int(train_row["label_5"])
        if int(r3["score_target"]) != gold:
            raise ValueError("rationale pair score differs from train")
        chosen_rationale = str(r3["rationale"])
        rejected_rationale = str(r2["rationale"])
        pair = {
            "schema_version": PAIR_SCHEMA_VERSION,
            "record_id": record_id,
            "gold_label": gold,
            "metric_id": str(train_row["metric_id"]),
            "language": str(train_row["language"]),
            "chosen": {"score": gold, "rationale": chosen_rationale},
            "rejected": {"score": gold, "rationale": rejected_rationale},
            "pair_type": "rationale_alignment",
            "pair_source": "rationale_control",
            "base_event_id": str(r3["base_event_id"]),
            "r3_reference_id": str(r3["arm_selected_reference_id"]),
            "r2_donor_reference_id": str(r2["arm_selected_reference_id"]),
            "r2_donor_event_id": str(r2["arm_rationale_source_event_id"]),
            "chosen_rationale_sha256": sha256_text(chosen_rationale),
            "rejected_rationale_sha256": sha256_text(rejected_rationale),
            "score_loss_mask": "off",
            "rationale_loss_mask": "rationale_content_tokens_only",
            "ordinal_cost": 0.0,
            "odpo_offset": 0.0,
        }
        pair["pair_hash"] = sha256_text(compact_json(pair))
        output.append(pair)
    record_ids = [str(pair["record_id"]) for pair in output]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("rationale pair contains duplicate record")
    return sorted(output, key=lambda pair: pair["record_id"])


def _load_raw_generation_indexes() -> tuple[
    dict[tuple[str, int], dict[str, Any]], dict[str, str]
]:
    output: dict[tuple[str, int], dict[str, Any]] = {}
    hashes = {}
    for seed in SEEDS:
        path = FAILURE_ROOT / f"private/seed{seed}/raw_generations.jsonl"
        report_path = FAILURE_ROOT / f"runs/seed{seed}_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        actual_hash = sha256_file(path)
        if actual_hash != str(report["raw_generations_sha256"]):
            raise ValueError(f"{path}: raw generation hash differs")
        rows = read_jsonl(path)
        if len(rows) != 2654:
            raise ValueError(f"{path}: row count differs")
        for row in rows:
            key = (str(row["record_id"]), seed)
            if key in output:
                raise ValueError(f"{path}: duplicate raw generation")
            output[key] = row
        hashes[f"seed{seed}"] = actual_hash
    return output, hashes


def build_actual_rationale_candidates(
    train_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    raw_by_record_seed: dict[tuple[str, int], dict[str, Any]],
    events_by_record: dict[str, list[dict[str, Any]]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    train_by_id = {str(row["record_id"]): row for row in train_rows}
    for row in failure_rows:
        record_id = str(row["record_id"])
        seed = int(row["generator_seed"])
        rationale = str(row.get("generated_rationale") or "")
        raw = raw_by_record_seed[(record_id, seed)]
        if (
            not bool(row["parse_success"])
            or int(row["generated_score"]) != int(row["gold_label"])
            or bool(row["forced_completion"])
            or bool(raw["forced_completion"])
            or str(raw["finish_reason"]) == "length"
            or int(raw["backend_generated_tokens"]) >= 256
            or not rationale.strip()
            or score_leakage(rationale, int(row["generated_score"]))
        ):
            continue
        human, anchor = choose_anchor(record_id, events_by_record)
        if (
            anchor is None
            or normalize_text(human) == normalize_text(rationale)
        ):
            continue
        model_token_count = len(
            tokenizer.encode(rationale, add_special_tokens=False)
        )
        human_token_count = len(list(anchor["rationale_token_ids"]))
        eligible[record_id].append(
            {
                "row": row,
                "raw": raw,
                "human": human,
                "anchor": anchor,
                "model_token_count": model_token_count,
                "human_token_count": human_token_count,
            }
        )

    output = []
    for record_id, values in eligible.items():
        selected = min(
            values,
            key=lambda value: (
                abs(
                    value["model_token_count"]
                    - value["human_token_count"]
                ),
                stable_hash(
                    "actual-rationale-candidate",
                    record_id,
                    value["row"]["generator_seed"],
                    sha256_text(value["row"]["generated_rationale"]),
                ),
            ),
        )
        row = selected["row"]
        source = train_by_id[record_id]
        gold = int(source["label_5"])
        human = str(selected["human"])
        model = str(row["generated_rationale"])
        candidate = {
            "schema_version": "exp54-actual-rationale-candidate-v1",
            "record_id": record_id,
            "gold_label": gold,
            "metric_id": str(source["metric_id"]),
            "language": str(source["language"]),
            "human_output": {"score": gold, "rationale": human},
            "model_output": {"score": gold, "rationale": model},
            "human_rationale_sha256": sha256_text(human),
            "model_rationale_sha256": sha256_text(model),
            "human_rationale_token_count": selected["human_token_count"],
            "model_rationale_token_count": selected["model_token_count"],
            "source_generator_seed": int(row["generator_seed"]),
            "source_checkpoint_sha256": str(
                row["generator_adapter_sha256"]
            ),
            "source_failure_row_sha256": sha256_text(compact_json(row)),
            "source_raw_generation_sha256": sha256_text(
                compact_json(selected["raw"])
            ),
            "forced_completion": False,
            "score_leakage": False,
            "preference_label_status": "PENDING_TWO_FAMILY_BLIND_REVIEW",
            "preference_training_allowed": False,
        }
        candidate["candidate_hash"] = sha256_text(compact_json(candidate))
        output.append(candidate)
    return sorted(output, key=lambda row: row["record_id"])


def _counts_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def summarize(
    *,
    hybrid: list[dict[str, Any]],
    synthetic: list[dict[str, Any]],
    rationale: list[dict[str, Any]],
    actual_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "hybrid_score_pairs": len(hybrid),
        "hybrid_unique_records": len(
            {str(row["record_id"]) for row in hybrid}
        ),
        "hybrid_pair_types": _counts_by(hybrid, "pair_type"),
        "hybrid_pair_sources": _counts_by(hybrid, "pair_source"),
        "hybrid_by_label": _counts_by(hybrid, "gold_label"),
        "hybrid_by_metric": _counts_by(hybrid, "metric_id"),
        "hybrid_by_language": _counts_by(hybrid, "language"),
        "hybrid_actual_support": _counts_by(
            hybrid, "actual_support_count"
        ),
        "hybrid_forced_evidence_pairs": sum(
            bool(row["forced_completion_evidence_any"]) for row in hybrid
        ),
        "hybrid_leakage_evidence_pairs": sum(
            bool(row["score_leakage_evidence_any"]) for row in hybrid
        ),
        "synthetic_score_pairs": len(synthetic),
        "synthetic_pair_types": _counts_by(synthetic, "pair_type"),
        "synthetic_by_label": _counts_by(synthetic, "gold_label"),
        "synthetic_by_metric": _counts_by(synthetic, "metric_id"),
        "synthetic_by_language": _counts_by(synthetic, "language"),
        "rationale_pairs": len(rationale),
        "rationale_unique_records": len(
            {str(row["record_id"]) for row in rationale}
        ),
        "rationale_by_label": _counts_by(rationale, "gold_label"),
        "rationale_by_metric": _counts_by(rationale, "metric_id"),
        "rationale_by_language": _counts_by(rationale, "language"),
        "actual_rationale_candidates": len(actual_candidates),
        "actual_rationale_candidate_by_label": _counts_by(
            actual_candidates, "gold_label"
        ),
        "actual_rationale_candidate_by_metric": _counts_by(
            actual_candidates, "metric_id"
        ),
        "actual_rationale_candidate_by_language": _counts_by(
            actual_candidates, "language"
        ),
        "actual_rationale_minimum_128_met": len(actual_candidates) >= 128,
    }


def build(*, write: bool) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if (
        protocol["split"] != "train"
        or protocol["source_arm"] != "R3"
        or protocol["preference_training_allowed"] is not False
        or protocol["dev_accessed"]
        or protocol["test_accessed"]
    ):
        raise ValueError("SORC-DPO pair protocol boundary differs")
    train_rows = load_train_rows()
    failure_report = json.loads(FAILURE_REPORT.read_text(encoding="utf-8"))
    if sha256_file(FAILURE_PATH) != str(
        failure_report["combined_private_sha256"]
    ):
        raise ValueError("actual failure-bank hash differs")
    failure_rows = read_jsonl(FAILURE_PATH)
    grouped = group_failure_evidence(train_rows, failure_rows)
    frozen_lock = json.loads(FROZEN_MANIFEST_LOCK.read_text(encoding="utf-8"))
    if not frozen_lock["manifest_frozen"] or frozen_lock["test_accessed"]:
        raise ValueError("materialized manifest freeze differs")
    events_by_record, paired_by_event, manifest_hashes = (
        _load_materialized_events(frozen_lock)
    )
    hybrid = build_hybrid_score_pairs(
        train_rows, grouped, events_by_record
    )
    synthetic = build_matched_synthetic_pairs(
        hybrid, train_rows, events_by_record
    )
    rationale = build_rationale_pairs(train_rows, paired_by_event)
    raw_by_record_seed, raw_hashes = _load_raw_generation_indexes()

    training_config = json.loads(
        TRAINING_CONFIG_PATH.read_text(encoding="utf-8")
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(training_config["model"]["local_path"]),
        local_files_only=True,
        trust_remote_code=False,
    )
    actual_candidates = build_actual_rationale_candidates(
        train_rows,
        failure_rows,
        raw_by_record_seed,
        events_by_record,
        tokenizer,
    )
    aggregate = summarize(
        hybrid=hybrid,
        synthetic=synthetic,
        rationale=rationale,
        actual_candidates=actual_candidates,
    )
    result = {
        "schema_version": "exp54-sorc-dpo-pair-candidate-report-v1",
        "status": (
            "SORC_DPO_PAIR_CANDIDATE_DRY_RUN_PASS"
            if not write
            else "SORC_DPO_PAIR_CANDIDATES_WRITTEN_NOT_FROZEN"
        ),
        "aggregate": aggregate,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "failure_bank_sha256": sha256_file(FAILURE_PATH),
        "frozen_manifest_lock_sha256": sha256_file(FROZEN_MANIFEST_LOCK),
        "materialized_manifest_hashes": manifest_hashes,
        "raw_generation_hashes": raw_hashes,
        "private_output_hashes": {},
        "actual_rationale_preference_labels_created": False,
        "rationale_blind_qualification_completed": False,
        "preference_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    if not write:
        return result

    private = OUTPUT_ROOT / "private"
    paths = {
        "score_pairs_hybrid": private / "score_pairs_hybrid.jsonl",
        "score_pairs_synthetic_seed42": (
            private / "score_pairs_synthetic_seed42.jsonl"
        ),
        "rationale_pairs_r3_r2": private / "rationale_pairs_r3_r2.jsonl",
        "actual_rationale_candidates": (
            private / "actual_rationale_candidates.jsonl"
        ),
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(path)
    write_jsonl(paths["score_pairs_hybrid"], hybrid)
    write_jsonl(paths["score_pairs_synthetic_seed42"], synthetic)
    write_jsonl(paths["rationale_pairs_r3_r2"], rationale)
    write_jsonl(paths["actual_rationale_candidates"], actual_candidates)
    result["private_output_hashes"] = {
        name: sha256_file(path) for name, path in paths.items()
    }
    report_path = OUTPUT_ROOT / "candidate_report.json"
    write_json(report_path, result)
    lock = {
        "schema_version": "exp54-sorc-dpo-pair-candidate-lock-v1",
        "status": "PAIR_CANDIDATES_NOT_FROZEN_TRAINING_NOT_ALLOWED",
        "protocol_sha256": result["protocol_sha256"],
        "failure_bank_sha256": result["failure_bank_sha256"],
        "frozen_manifest_lock_sha256": result[
            "frozen_manifest_lock_sha256"
        ],
        "private_output_hashes": result["private_output_hashes"],
        "candidate_report_sha256": sha256_file(report_path),
        "preference_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    write_json(OUTPUT_ROOT / "candidate_lock.json", lock)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            build(write=args.write),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
