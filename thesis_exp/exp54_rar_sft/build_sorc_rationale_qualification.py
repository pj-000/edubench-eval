"""Build the train-only blind qualification package for SORC-DPO rationale pairs.

The package is deliberately created before any evaluator is called. Selection
uses only frozen pair metadata and train metadata; it never reads dev, test, or
judge outputs. Each selected R3-vs-R2 pair is presented in both A/B orders for
score-blind and score-visible review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import (
    compact_json,
    load_train_rows,
    read_jsonl,
    sha256_bytes,
    sha256_file,
)


SCHEMA_VERSION = "exp54-sorc-rationale-qualification-v1"
SELECTOR_SEED = "exp54-sorc-rationale-qualification-v1|20260729"
QUOTAS = {"L12": 24, "L3": 32, "L45": 64}
EXPECTED_PAIRS = sum(QUOTAS.values())
EXPECTED_PRESENTATIONS_PER_STAGE = EXPECTED_PAIRS * 2
EXPECTED_METRICS = 12
EXPECTED_LANGUAGES = {"en", "zh"}
PAIR_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/preference_pairs"
)
PAIR_PATH = PAIR_ROOT / "private/rationale_pairs_r3_r2.jsonl"
PAIR_LOCK_PATH = PAIR_ROOT / "candidate_lock.json"
OUTPUT_ROOT = PAIR_ROOT / "rationale_qualification"
PROMPT_ROOT = REPO_ROOT / "thesis_exp/exp54_rar_sft/prompts"
SCHEMA_ROOT = REPO_ROOT / "thesis_exp/exp54_rar_sft/schemas"
PROMPTS = {
    "score_blind": PROMPT_ROOT / "rationale_audit_score_blind_v2.txt",
    "score_visible": PROMPT_ROOT / "rationale_audit_score_visible_v2.txt",
}
SCHEMAS = {
    "score_blind": (
        SCHEMA_ROOT / "rationale_audit_score_blind_v2.schema.json"
    ),
    "score_visible": (
        SCHEMA_ROOT / "rationale_audit_score_visible_v2.schema.json"
    ),
}


def stable_hash(*parts: Any) -> str:
    payload = compact_json(list(parts)).encode("utf-8")
    return sha256_bytes(payload)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    payload = b"".join(
        (compact_json(row) + "\n").encode("utf-8") for row in materialized
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return {
        "bytes": len(payload),
        "rows": len(materialized),
        "sha256": sha256_bytes(payload),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def label_band(label: int) -> str:
    if label in {1, 2}:
        return "L12"
    if label == 3:
        return "L3"
    if label in {4, 5}:
        return "L45"
    raise ValueError("label outside 1-5")


def _rank_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        stable_hash(
            SELECTOR_SEED,
            row["pair_hash"],
            row["record_id"],
        ),
        str(row["record_id"]),
    )


def choose_balanced(
    *,
    candidates: list[dict[str, Any]],
    quota: int,
    already_selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    remaining = list(candidates)
    chosen: list[dict[str, Any]] = []
    while len(chosen) < quota:
        if not remaining:
            raise ValueError("rationale qualification quota cannot be met")
        current = already_selected + chosen
        metric_counts = Counter(str(row["metric_id"]) for row in current)
        language_counts = Counter(str(row["language"]) for row in current)
        joint_counts = Counter(
            (str(row["metric_id"]), str(row["language"]))
            for row in current
        )
        selected = min(
            remaining,
            key=lambda row: (
                joint_counts[
                    (str(row["metric_id"]), str(row["language"]))
                ],
                metric_counts[str(row["metric_id"])],
                language_counts[str(row["language"])],
                *_rank_key(row),
            ),
        )
        chosen.append(selected)
        remaining.remove(selected)
    return chosen


def validate_source_pairs(rows: list[dict[str, Any]]) -> None:
    if len(rows) < EXPECTED_PAIRS:
        raise ValueError("too few rationale pairs")
    record_ids: set[str] = set()
    pair_hashes: set[str] = set()
    for row in rows:
        record_id = str(row.get("record_id") or "")
        pair_hash = str(row.get("pair_hash") or "")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not record_id or record_id in record_ids:
            raise ValueError("rationale pair record IDs are empty or duplicated")
        if not pair_hash or pair_hash in pair_hashes:
            raise ValueError("rationale pair hashes are empty or duplicated")
        record_ids.add(record_id)
        pair_hashes.add(pair_hash)
        if row.get("pair_type") != "rationale_alignment":
            raise ValueError("unexpected rationale pair type")
        if row.get("pair_source") != "rationale_control":
            raise ValueError("unexpected rationale pair source")
        if row.get("score_loss_mask") != "off":
            raise ValueError("rationale pair has score supervision")
        if row.get("rationale_loss_mask") != "rationale_content_tokens_only":
            raise ValueError("rationale pair mask differs")
        if not isinstance(chosen, dict) or not isinstance(rejected, dict):
            raise ValueError("rationale pair candidates are malformed")
        if int(chosen["score"]) != int(rejected["score"]):
            raise ValueError("rationale pair changes score")
        if int(chosen["score"]) != int(row["gold_label"]):
            raise ValueError("rationale pair score differs from label")
        chosen_text = str(chosen.get("rationale") or "")
        rejected_text = str(rejected.get("rationale") or "")
        if not chosen_text or not rejected_text or chosen_text == rejected_text:
            raise ValueError("rationale contrast is empty")
        if sha256_bytes(chosen_text.encode("utf-8")) != str(
            row["chosen_rationale_sha256"]
        ):
            raise ValueError("chosen rationale hash differs")
        if sha256_bytes(rejected_text.encode("utf-8")) != str(
            row["rejected_rationale_sha256"]
        ):
            raise ValueError("rejected rationale hash differs")


def select_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_source_pairs(rows)
    selected: list[dict[str, Any]] = []
    for band, quota in QUOTAS.items():
        candidates = [
            row for row in rows if label_band(int(row["gold_label"])) == band
        ]
        selected.extend(
            choose_balanced(
                candidates=candidates,
                quota=quota,
                already_selected=selected,
            )
        )
    selected.sort(key=_rank_key)
    if len(selected) != EXPECTED_PAIRS:
        raise ValueError("selected pair count differs")
    if len({str(row["record_id"]) for row in selected}) != EXPECTED_PAIRS:
        raise ValueError("selected records are not unique")
    if len({str(row["metric_id"]) for row in selected}) != EXPECTED_METRICS:
        raise ValueError("selected pairs do not cover all metrics")
    if {str(row["language"]) for row in selected} != EXPECTED_LANGUAGES:
        raise ValueError("selected pairs do not cover both languages")
    return selected


def _task_candidate(
    *,
    rationale: str,
    score: int,
    score_visible: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {"rationale": rationale}
    if score_visible:
        result["score"] = score
    return result


def build_package(
    *,
    train_rows: list[dict[str, Any]],
    source_pairs: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    train_by_id = {str(row["record_id"]): row for row in train_rows}
    if len(train_by_id) != len(train_rows):
        raise ValueError("train record IDs are duplicated")
    selected = select_pairs(source_pairs)
    sample_manifest: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    tasks = {"score_blind": [], "score_visible": []}

    for pair in selected:
        record_id = str(pair["record_id"])
        source = train_by_id.get(record_id)
        if source is None:
            raise ValueError("rationale pair record is absent from train")
        for field in ("metric_id", "language"):
            if str(source[field]) != str(pair[field]):
                raise ValueError(f"pair {field} differs from train")
        if int(source["label_5"]) != int(pair["gold_label"]):
            raise ValueError("pair label differs from train")

        pair_id = stable_hash(
            SCHEMA_VERSION,
            pair["pair_hash"],
            record_id,
        )
        sample_manifest.append(
            {
                "pair_id": pair_id,
                "record_id": record_id,
                "source_pair_hash": str(pair["pair_hash"]),
                "gold_label": int(pair["gold_label"]),
                "metric_id": str(pair["metric_id"]),
                "language": str(pair["language"]),
                "question": source["question"],
                "answer": source["answer"],
                "metric": str(
                    source.get("metric_canonical")
                    or source.get("metric_raw")
                    or source["metric_id"]
                ),
                "rubric": source["rubric"],
                "r3_aligned_rationale": str(pair["chosen"]["rationale"]),
                "r2_shuffled_rationale": str(pair["rejected"]["rationale"]),
            }
        )
        swap_first = int(stable_hash(SELECTOR_SEED, pair_id)[:2], 16) % 2
        for orientation_index in (0, 1):
            r3_is_a = bool(swap_first ^ orientation_index)
            presentation_id = stable_hash(
                SCHEMA_VERSION,
                pair_id,
                orientation_index,
            )
            answer_key.append(
                {
                    "presentation_id": presentation_id,
                    "pair_id": pair_id,
                    "orientation_index": orientation_index,
                    "a_source": "R3_ALIGNED" if r3_is_a else "R2_SHUFFLED",
                    "b_source": "R2_SHUFFLED" if r3_is_a else "R3_ALIGNED",
                }
            )
            rationale_a = str(
                pair["chosen" if r3_is_a else "rejected"]["rationale"]
            )
            rationale_b = str(
                pair["rejected" if r3_is_a else "chosen"]["rationale"]
            )
            common = {
                "presentation_id": presentation_id,
                "question": source["question"],
                "answer": source["answer"],
                "metric_id": str(pair["metric_id"]),
                "metric": str(
                    source.get("metric_canonical")
                    or source.get("metric_raw")
                    or source["metric_id"]
                ),
                "rubric": source["rubric"],
            }
            for stage, score_visible in (
                ("score_blind", False),
                ("score_visible", True),
            ):
                tasks[stage].append(
                    {
                        **common,
                        "candidate_a": _task_candidate(
                            rationale=rationale_a,
                            score=int(pair["gold_label"]),
                            score_visible=score_visible,
                        ),
                        "candidate_b": _task_candidate(
                            rationale=rationale_b,
                            score=int(pair["gold_label"]),
                            score_visible=score_visible,
                        ),
                    }
                )

    for stage in tasks:
        if len(tasks[stage]) != EXPECTED_PRESENTATIONS_PER_STAGE:
            raise ValueError(f"{stage} presentation count differs")
        if len(
            {str(row["presentation_id"]) for row in tasks[stage]}
        ) != EXPECTED_PRESENTATIONS_PER_STAGE:
            raise ValueError(f"{stage} presentation IDs are duplicated")
    if Counter(row["presentation_id"] for row in answer_key) != Counter(
        row["presentation_id"] for row in tasks["score_blind"]
    ):
        raise ValueError("answer-key closure differs")
    return {
        "sample_manifest": sample_manifest,
        "answer_key": answer_key,
        "score_blind_tasks": tasks["score_blind"],
        "score_visible_tasks": tasks["score_visible"],
    }


def summarize(package: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    samples = package["sample_manifest"]
    return {
        "selected_pairs": len(samples),
        "presentations_per_stage": len(package["score_blind_tasks"]),
        "stages": 2,
        "total_presentations_per_evaluator_family": (
            len(package["score_blind_tasks"])
            + len(package["score_visible_tasks"])
        ),
        "evaluator_families_required": 2,
        "label_band_counts": dict(
            sorted(
                Counter(
                    label_band(int(row["gold_label"])) for row in samples
                ).items()
            )
        ),
        "language_counts": dict(
            sorted(Counter(str(row["language"]) for row in samples).items())
        ),
        "metric_count": len({str(row["metric_id"]) for row in samples}),
        "orientation_counts": dict(
            sorted(
                Counter(
                    str(row["a_source"])
                    for row in package["answer_key"]
                ).items()
            )
        ),
    }


def build(*, write: bool) -> dict[str, Any]:
    if not PAIR_PATH.exists():
        raise FileNotFoundError(PAIR_PATH)
    if not PAIR_LOCK_PATH.exists():
        raise FileNotFoundError(PAIR_LOCK_PATH)
    pair_lock = json.loads(PAIR_LOCK_PATH.read_text(encoding="utf-8"))
    expected_pair_hash = str(
        pair_lock["private_output_hashes"]["rationale_pairs_r3_r2"]
    )
    if sha256_file(PAIR_PATH) != expected_pair_hash:
        raise ValueError("rationale pair file differs from candidate lock")
    for path in (*PROMPTS.values(), *SCHEMAS.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    package = build_package(
        train_rows=load_train_rows(),
        source_pairs=read_jsonl(PAIR_PATH),
    )
    result = {
        "schema_version": f"{SCHEMA_VERSION}-candidate-report",
        "status": (
            "SORC_RATIONALE_QUALIFICATION_DRY_RUN_PASS"
            if not write
            else "SORC_RATIONALE_QUALIFICATION_PACKAGE_WRITTEN_NOT_RUN"
        ),
        "aggregate": summarize(package),
        "source_hashes": {
            "train": sha256_file(
                REPO_ROOT
                / "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl"
            ),
            "rationale_pairs": expected_pair_hash,
            "pair_candidate_lock": sha256_file(PAIR_LOCK_PATH),
            "builder_source": sha256_file(Path(__file__)),
            **{
                f"{stage}_prompt": sha256_file(PROMPTS[stage])
                for stage in PROMPTS
            },
            **{
                f"{stage}_schema": sha256_file(SCHEMAS[stage])
                for stage in SCHEMAS
            },
        },
        "private_output_hashes": {},
        "selection_uses_evaluator_results": False,
        "rationale_blind_qualification_completed": False,
        "actual_rationale_preference_labels_created": False,
        "p3_preference_training_allowed": False,
        "preference_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    if not write:
        return result

    private_root = OUTPUT_ROOT / "private"
    paths = {
        name: private_root / f"{name}.jsonl" for name in package
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(path)
    result["private_output_hashes"] = {
        name: write_jsonl(paths[name], rows)
        for name, rows in package.items()
    }
    report_path = OUTPUT_ROOT / "candidate_report.json"
    write_json(report_path, result)
    lock = {
        "schema_version": f"{SCHEMA_VERSION}-candidate-lock",
        "status": "QUALIFICATION_PACKAGE_NOT_RUN_TRAINING_NOT_ALLOWED",
        "source_hashes": result["source_hashes"],
        "private_output_hashes": result["private_output_hashes"],
        "candidate_report_sha256": sha256_file(report_path),
        "rationale_blind_qualification_completed": False,
        "p3_preference_training_allowed": False,
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
