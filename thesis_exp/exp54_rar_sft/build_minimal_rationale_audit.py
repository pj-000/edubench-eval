"""Build the private 40-row Exp54 rationale blind-audit package.

The selector reads only frozen dev metadata. It does not inspect any generated
score, rationale, forced-completion flag, arm identity, or judge preference.
Generated outputs are joined only after the 40 row positions are fixed.
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


ARMS = ("R1", "R2", "R3")
SEEDS = (42, 43, 44)
COMPARISONS = (
    ("primary", "R3", "R2"),
    ("secondary", "R3", "R1"),
)
SELECTOR_SEED = "exp54-minimal-rationale-audit-v2|20260728"
EXPECTED_DEV_ROWS = 664
EXPECTED_LOW_ROWS = 20
QUOTAS = {"L3": 8, "L45": 12}
DEFAULT_DEV = (
    REPO_ROOT / "thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl"
)
DEFAULT_DEV_RUNS = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/dev_runs_vllm"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "rationale_blind_audit"
)


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_hash(*parts: Any) -> str:
    return sha256_bytes("|".join(str(part) for part in parts).encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        (compact_json(row) + "\n").encode("utf-8") for row in rows
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "rows": len(payload.splitlines()),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def rank_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        stable_hash(
            SELECTOR_SEED,
            row["record_id"],
            row["row_position"],
        ),
        str(row["record_id"]),
        int(row["row_position"]),
    )


def choose_band(
    *,
    candidates: list[dict[str, Any]],
    quota: int,
    already_selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    remaining = list(candidates)
    chosen: list[dict[str, Any]] = []
    while len(chosen) < quota:
        if not remaining:
            raise ValueError("audit sample band cannot satisfy its quota")
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
                metric_counts[str(row["metric_id"])],
                language_counts[str(row["language"])],
                joint_counts[
                    (str(row["metric_id"]), str(row["language"]))
                ],
                *rank_key(row),
            ),
        )
        chosen.append(selected)
        remaining.remove(selected)
    return chosen


def select_rows(dev_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(dev_rows) != EXPECTED_DEV_ROWS:
        raise ValueError("dev row count differs")
    prepared = []
    seen_ids = set()
    for row_position, source in enumerate(dev_rows):
        row = {
            "record_id": str(source["record_id"]),
            "row_position": row_position,
            "label_5": int(source["label_5"]),
            "metric_id": str(source["metric_id"]),
            "language": str(source["language"]),
        }
        if not row["record_id"] or row["record_id"] in seen_ids:
            raise ValueError("dev record IDs are empty or duplicated")
        seen_ids.add(row["record_id"])
        prepared.append(row)

    low = [row for row in prepared if row["label_5"] <= 2]
    if len(low) != EXPECTED_LOW_ROWS:
        raise ValueError("low-score dev count differs")
    selected = sorted(low, key=lambda row: int(row["row_position"]))
    for band, quota in QUOTAS.items():
        candidates = [
            row
            for row in prepared
            if (
                (band == "L3" and row["label_5"] == 3)
                or (band == "L45" and row["label_5"] >= 4)
            )
        ]
        selected.extend(
            choose_band(
                candidates=candidates,
                quota=quota,
                already_selected=selected,
            )
        )
    selected = sorted(selected, key=lambda row: int(row["row_position"]))
    if len(selected) != 40 or len({row["record_id"] for row in selected}) != 40:
        raise ValueError("minimal audit sample is not 40 unique rows")
    if len({row["metric_id"] for row in selected}) != 12:
        raise ValueError("minimal audit sample does not cover all 12 metrics")
    if len({row["language"] for row in selected}) != 2:
        raise ValueError("minimal audit sample does not cover both languages")
    return selected


def prediction_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != EXPECTED_DEV_ROWS:
        raise ValueError(f"{path}: prediction row count differs")
    index = {}
    for row_position, row in enumerate(rows):
        record_id = str(row["record_id"])
        if record_id in index:
            raise ValueError(f"{path}: duplicate record ID")
        if row.get("parse_success") is not True:
            raise ValueError(f"{path}: unparsed prediction")
        prediction = row.get("prediction")
        if (
            not isinstance(prediction, dict)
            or not isinstance(prediction.get("score"), int)
            or not isinstance(prediction.get("rationale"), str)
        ):
            raise ValueError(f"{path}: malformed structured prediction")
        index[record_id] = {
            "row_position": row_position,
            "score": int(prediction["score"]),
            "rationale": str(prediction["rationale"]),
            "forced_completion": bool(row.get("forced_completion")),
        }
    return index


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    dev_rows = read_jsonl(args.dev)
    selected = select_rows(dev_rows)
    dev_by_id = {str(row["record_id"]): row for row in dev_rows}

    predictions: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = (
                args.dev_runs
                / arm.lower()
                / f"seed{seed}"
                / "epoch3"
                / "predictions.jsonl"
            )
            predictions[(arm, seed)] = prediction_index(path)
    expected_vector = [str(row["record_id"]) for row in dev_rows]
    for index in predictions.values():
        if list(index) != expected_vector:
            raise ValueError("prediction record order differs from dev")

    private = args.output / "private"
    sample_manifest = []
    for selected_row in selected:
        source = dev_by_id[selected_row["record_id"]]
        sample_manifest.append(
            {
                **selected_row,
                "question": str(source["question"]),
                "answer": str(source["answer"]),
                "rubric": source["rubric"],
            }
        )

    score_blind_tasks = []
    score_visible_tasks = []
    answer_key = []
    pair_count = Counter()
    forced_counts = Counter()
    for comparison, target_arm, comparator_arm in COMPARISONS:
        for seed in SEEDS:
            target = predictions[(target_arm, seed)]
            comparator = predictions[(comparator_arm, seed)]
            for selected_row in selected:
                record_id = selected_row["record_id"]
                context = dev_by_id[record_id]
                pair_id = stable_hash(
                    "exp54-rationale-pair-v2",
                    comparison,
                    seed,
                    record_id,
                )
                pair_count[comparison] += 1
                forced_counts[
                    (
                        comparison,
                        target_arm,
                        target[record_id]["forced_completion"],
                    )
                ] += 1
                forced_counts[
                    (
                        comparison,
                        comparator_arm,
                        comparator[record_id]["forced_completion"],
                    )
                ] += 1
                base_swapped = int(pair_id[:2], 16) % 2 == 1
                arms = (target_arm, comparator_arm)
                for orientation in (0, 1):
                    swapped = base_swapped ^ bool(orientation)
                    a_arm, b_arm = (arms[::-1] if swapped else arms)
                    candidate_a = predictions[(a_arm, seed)][record_id]
                    candidate_b = predictions[(b_arm, seed)][record_id]
                    presentation_id = stable_hash(
                        pair_id, "orientation", orientation
                    )
                    common = {
                        "presentation_id": presentation_id,
                        "question": str(context["question"]),
                        "answer": str(context["answer"]),
                        "metric_id": str(context["metric_id"]),
                        "metric": str(
                            context.get("metric_canonical")
                            or context.get("metric_raw")
                            or context["metric_id"]
                        ),
                        "rubric": context["rubric"],
                    }
                    score_blind_tasks.append(
                        {
                            **common,
                            "stage": "score_blind",
                            "candidate_a": {
                                "rationale": candidate_a["rationale"]
                            },
                            "candidate_b": {
                                "rationale": candidate_b["rationale"]
                            },
                        }
                    )
                    score_visible_tasks.append(
                        {
                            **common,
                            "stage": "score_visible",
                            "candidate_a": {
                                "score": candidate_a["score"],
                                "rationale": candidate_a["rationale"],
                            },
                            "candidate_b": {
                                "score": candidate_b["score"],
                                "rationale": candidate_b["rationale"],
                            },
                        }
                    )
                    answer_key.append(
                        {
                            "presentation_id": presentation_id,
                            "pair_id": pair_id,
                            "comparison": comparison,
                            "orientation": orientation,
                            "record_id": record_id,
                            "row_position": selected_row["row_position"],
                            "label_5": selected_row["label_5"],
                            "seed": seed,
                            "candidate_a_arm": a_arm,
                            "candidate_b_arm": b_arm,
                            "candidate_a_forced": candidate_a[
                                "forced_completion"
                            ],
                            "candidate_b_forced": candidate_b[
                                "forced_completion"
                            ],
                        }
                    )

    bindings = {
        "sample_manifest": write_jsonl(
            private / "sample_manifest.jsonl", sample_manifest
        ),
        "score_blind_tasks": write_jsonl(
            private / "score_blind_tasks.jsonl", score_blind_tasks
        ),
        "score_visible_tasks": write_jsonl(
            private / "score_visible_tasks.jsonl", score_visible_tasks
        ),
        "answer_key": write_jsonl(
            private / "answer_key.jsonl", answer_key
        ),
    }
    label_counts = Counter(int(row["label_5"]) for row in selected)
    metric_counts = Counter(str(row["metric_id"]) for row in selected)
    language_counts = Counter(str(row["language"]) for row in selected)
    report = {
        "schema_version": "exp54-minimal-rationale-audit-package-v2",
        "status": "MINIMAL_RATIONALE_AUDIT_PACKAGE_READY",
        "selector": {
            "version": "exp54-minimal-rationale-selector-v2",
            "seed": SELECTOR_SEED,
            "low_score_rule": "include every label_5 <= 2 row",
            "band_quotas": QUOTAS,
            "candidate_rank": (
                "min(current metric count, current language count, current "
                "metric-language count, SHA256(seed|record_id|row_position), "
                "record_id, row_position)"
            ),
            "band_order": ["L3", "L45"],
            "final_order": "ascending original dev row_position",
            "selection_reads_model_outputs": False,
        },
        "sample": {
            "unique_rows": len(selected),
            "label_counts": {
                str(key): label_counts[key] for key in sorted(label_counts)
            },
            "metric_counts": {
                key: metric_counts[key] for key in sorted(metric_counts)
            },
            "language_counts": {
                key: language_counts[key] for key in sorted(language_counts)
            },
            "all_12_metrics_covered": len(metric_counts) == 12,
            "both_languages_covered": len(language_counts) == 2,
        },
        "tasks": {
            "pair_instances_before_orientation": {
                key: int(value) for key, value in pair_count.items()
            },
            "presentations_per_stage": len(score_blind_tasks),
            "stages": ["score_blind", "score_visible"],
            "total_presentations": (
                len(score_blind_tasks) + len(score_visible_tasks)
            ),
            "orientations_per_pair": 2,
        },
        "private_bindings": bindings,
        "forced_completion_summary": {
            f"{comparison}|{arm}|forced={str(forced).lower()}": int(count)
            for (comparison, arm, forced), count in sorted(
                forced_counts.items()
            )
        },
        "evaluator_families_required": 2,
        "evaluator_calls_completed": False,
        "test_accessed": False,
        "preference_training_started": False,
    }
    write_json(args.output / "candidate_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--dev-runs", type=Path, default=DEFAULT_DEV_RUNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    report = build_package(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
