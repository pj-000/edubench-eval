"""CPU-only Stage-0 audit and group split for Exp62.

The official SummEval annotation release contains the candidate summaries and
three independent expert ratings.  It does not contain the corresponding
CNN/DailyMail source articles.  The frozen core experiment therefore uses only
the source-free coherence and fluency dimensions.  Consistency and relevance
must not be enabled unless their source articles are independently recovered
and locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp62_summeval_routing_confirmation import (
    DIMENSIONS,
    LABELS,
    NUM_RATERS,
    OUTPUT_ROOT,
)


EXPECTED_ANNOTATION_SHA256 = "f0d4166e0cdeb439b387c4449634b067b7e9f721b9ef4c2b722f6b7550d3d6ab"
EXPECTED_RAW_ROWS = 1600
EXPECTED_GROUPS = 100
EXPECTED_MODELS = 16
EXPECTED_ROWS_PER_GROUP = 16
EXPECTED_GROUP_COUNTS = {"train": 70, "dev": 15, "test": 15}
EXPECTED_EXPANDED_ROWS = {"train": 2240, "dev": 480, "test": 480}
SPLIT_SEARCH_SEEDS = 10_000


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized_text(value: str) -> str:
    return " ".join(value.lower().split())


def require_all(gates: dict[str, bool], context: str) -> None:
    failures = sorted(name for name, passed in gates.items() if not passed)
    if failures:
        raise RuntimeError(f"{context} failed closed: {', '.join(failures)}")


def quantized_mean_label(scores: Iterable[int]) -> int:
    values = tuple(int(value) for value in scores)
    if len(values) != NUM_RATERS or any(value not in LABELS for value in values):
        raise ValueError("Exp62 requires exactly three integer ratings in [1, 5]")
    return math.floor(sum(values) / len(values) + 0.5)


def empirical_target(scores: Iterable[int]) -> tuple[float, ...]:
    values = tuple(int(value) for value in scores)
    if len(values) != NUM_RATERS or any(value not in LABELS for value in values):
        raise ValueError("Exp62 requires exactly three integer ratings in [1, 5]")
    counts = Counter(values)
    return tuple(counts[label] / NUM_RATERS for label in LABELS)


def _expert_scores(item: dict[str, Any], dimension: str) -> tuple[int, ...]:
    annotations = item.get("expert_annotations")
    if not isinstance(annotations, list) or len(annotations) != NUM_RATERS:
        raise RuntimeError("SummEval row does not contain exactly three expert annotations")
    try:
        scores = tuple(annotation[dimension] for annotation in annotations)
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"missing expert dimension: {dimension}") from error
    if any(type(value) is not int or value not in LABELS for value in scores):
        raise RuntimeError(f"invalid expert scores for {dimension}: {scores}")
    return scores


def load_annotations(path: Path, *, verify_hash: bool = True) -> list[dict[str, Any]]:
    if verify_hash and sha256_file(path) != EXPECTED_ANNOTATION_SHA256:
        raise RuntimeError("SummEval annotation SHA256 mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != EXPECTED_RAW_ROWS:
        raise RuntimeError(f"SummEval has {len(rows)} rows instead of {EXPECTED_RAW_ROWS}")
    return rows


def audit_raw_annotations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    group_models: defaultdict[str, list[str]] = defaultdict(list)
    group_paths: defaultdict[str, set[str]] = defaultdict(set)
    summary_groups: defaultdict[str, set[str]] = defaultdict(set)
    example_keys: list[str] = []
    model_ids: set[str] = set()
    empty_summaries = 0
    for item in rows:
        group = str(item.get("id", ""))
        model = str(item.get("model_id", ""))
        path = str(item.get("filepath", ""))
        summary = str(item.get("decoded", ""))
        if not group or not model or not path:
            raise RuntimeError("SummEval row is missing id, model_id, or filepath")
        if not summary.strip():
            empty_summaries += 1
        for dimension in ("coherence", "fluency", "consistency", "relevance"):
            _expert_scores(item, dimension)
        example_keys.append(f"{group}:{model}")
        model_ids.add(model)
        group_models[group].append(model)
        group_paths[group].add(path)
        summary_groups[normalized_text(summary)].add(group)

    expected_model_set = set(model_ids)
    crossing_summaries = {
        summary: sorted(groups)
        for summary, groups in summary_groups.items()
        if len(groups) > 1
    }
    gates = {
        "raw_row_count": len(rows) == EXPECTED_RAW_ROWS,
        "unique_example_keys": len(set(example_keys)) == EXPECTED_RAW_ROWS,
        "group_count": len(group_models) == EXPECTED_GROUPS,
        "model_count": len(model_ids) == EXPECTED_MODELS,
        "sixteen_models_per_group": all(
            len(values) == EXPECTED_ROWS_PER_GROUP for values in group_models.values()
        ),
        "same_model_set_per_group": all(
            set(values) == expected_model_set for values in group_models.values()
        ),
        "one_source_path_per_group": all(len(values) == 1 for values in group_paths.values()),
        "no_empty_summary": empty_summaries == 0,
        "no_summary_crosses_source_group": not crossing_summaries,
    }
    require_all(gates, "SummEval raw-data audit")
    return {
        "checks": gates,
        "raw_rows": len(rows),
        "source_groups": len(group_models),
        "model_ids": sorted(model_ids),
        "empty_summaries": empty_summaries,
        "normalized_summary_duplicates_within_group": len(rows) - len(summary_groups),
        "normalized_summaries_crossing_group": len(crossing_summaries),
    }


def expand_records(
    rows: list[dict[str, Any]], *, expected_raw_rows: int | None = EXPECTED_RAW_ROWS
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in rows:
        group = str(item["id"])
        model = str(item["model_id"])
        summary = str(item["decoded"])
        for dimension in DIMENSIONS:
            scores = _expert_scores(item, dimension)
            target = empirical_target(scores)
            hard_label = quantized_mean_label(scores)
            records.append(
                {
                    "record_id": f"summeval:{group}:{model}:{dimension}",
                    "group_id": group,
                    "source_filepath": str(item["filepath"]),
                    "model_id": model,
                    "dimension": dimension,
                    "summary": summary,
                    "summary_sha256": sha256_bytes(normalized_text(summary).encode("utf-8")),
                    "expert_scores": list(scores),
                    "soft_target": list(target),
                    "human_mean": sum(scores) / NUM_RATERS,
                    "hard_label": hard_label,
                    "target_sha256": sha256_bytes(stable_json(list(scores)).encode("utf-8")),
                }
            )
    if expected_raw_rows is not None and len(records) != expected_raw_rows * len(DIMENSIONS):
        raise RuntimeError("unexpected Exp62 expanded row count")
    if len({item["record_id"] for item in records}) != len(records):
        raise RuntimeError("duplicate Exp62 record ID")
    return records


def _distribution(records: list[dict[str, Any]], groups: set[str]) -> dict[str, list[int]]:
    counts = Counter(
        (item["dimension"], item["hard_label"])
        for item in records
        if item["group_id"] in groups
    )
    return {
        dimension: [counts[(dimension, label)] for label in LABELS]
        for dimension in DIMENSIONS
    }


def choose_group_split(records: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Choose one deterministic label-balanced 70/15/15 article split.

    The search uses labels only, never model predictions.  A candidate is
    eligible only if every dimension/split contains all five hard labels.  We
    then minimize the largest absolute class-proportion deviation from the
    full corpus, with the integer seed as a deterministic tie-breaker.
    """

    groups = sorted({item["group_id"] for item in records})
    global_distribution = _distribution(records, set(groups))
    best: tuple[float, int, dict[str, set[str]]] | None = None
    for seed in range(SPLIT_SEARCH_SEEDS):
        shuffled = list(groups)
        random.Random(seed).shuffle(shuffled)
        candidate = {
            "train": set(shuffled[:70]),
            "dev": set(shuffled[70:85]),
            "test": set(shuffled[85:]),
        }
        score = 0.0
        eligible = True
        for split, split_groups in candidate.items():
            distribution = _distribution(records, split_groups)
            for dimension in DIMENSIONS:
                if any(value == 0 for value in distribution[dimension]):
                    eligible = False
                    break
                split_total = sum(distribution[dimension])
                global_total = sum(global_distribution[dimension])
                for observed, global_value in zip(
                    distribution[dimension], global_distribution[dimension]
                ):
                    score = max(
                        score,
                        abs(observed / split_total - global_value / global_total),
                    )
            if not eligible:
                break
        if eligible and (best is None or (score, seed) < (best[0], best[1])):
            best = (score, seed, candidate)
    if best is None:
        raise RuntimeError("no eligible Exp62 group split found")
    score, seed, assignment = best
    return assignment, {
        "search_seeds": SPLIT_SEARCH_SEEDS,
        "selected_seed": seed,
        "maximum_absolute_class_proportion_deviation": score,
        "hard_label_counts": {
            split: _distribution(records, values) for split, values in assignment.items()
        },
    }


def serialize_manifest(records: list[dict[str, Any]], assignment: dict[str, set[str]]) -> str:
    group_to_split = {
        group: split for split, groups in assignment.items() for group in groups
    }
    manifest: list[dict[str, Any]] = []
    for item in sorted(records, key=lambda value: value["record_id"]):
        manifest.append(
            {
                "record_id": item["record_id"],
                "group_id": item["group_id"],
                "model_id": item["model_id"],
                "dimension": item["dimension"],
                "summary_sha256": item["summary_sha256"],
                "target_sha256": item["target_sha256"],
                "hard_label": item["hard_label"],
                "split": group_to_split[item["group_id"]],
            }
        )
    return "".join(stable_json(item) + "\n" for item in manifest)


def build_report(
    annotation_path: Path,
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    raw_audit: dict[str, Any],
    assignment: dict[str, set[str]],
    split_audit: dict[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    counts = Counter(
        (item["dimension"], score)
        for item in records
        for score in item["expert_scores"]
    )
    unanimous = Counter(
        item["dimension"]
        for item in records
        if len(set(item["expert_scores"])) == 1
    )
    expanded_counts = Counter(
        next(split for split, groups in assignment.items() if item["group_id"] in groups)
        for item in records
    )
    split_gates = {
        "disjoint_groups": not (
            assignment["train"] & assignment["dev"]
            or assignment["train"] & assignment["test"]
            or assignment["dev"] & assignment["test"]
        ),
        "all_groups_assigned": len(set().union(*assignment.values())) == EXPECTED_GROUPS,
        "expected_group_counts": {
            split: len(groups) for split, groups in assignment.items()
        } == EXPECTED_GROUP_COUNTS,
        "expected_expanded_rows": dict(expanded_counts) == EXPECTED_EXPANDED_ROWS,
        "all_labels_in_each_dimension_and_split": all(
            all(value > 0 for value in split_audit["hard_label_counts"][split][dimension])
            for split in ("train", "dev", "test")
            for dimension in DIMENSIONS
        ),
    }
    require_all(split_gates, "Exp62 split audit")
    return {
        "status": "EXP62_STAGE0_PASS_READY_TO_FREEZE_PROTOCOL",
        "source": {
            "annotation_path": str(annotation_path.resolve()),
            "annotation_sha256": sha256_file(annotation_path),
            "official_annotation_url": (
                "https://storage.googleapis.com/sfr-summarization-repo-research/"
                "model_annotations.aligned.jsonl"
            ),
            "official_repository": "https://github.com/Yale-LILY/SummEval",
            "official_repository_commit": "81b59ad53d63cb6009764240853c91235a44e238",
        },
        "scope": {
            "dimensions": list(DIMENSIONS),
            "reason": (
                "The released annotation JSONL contains candidate summaries but not source "
                "articles. Coherence and fluency are source-free; consistency and relevance "
                "are excluded from the frozen core experiment."
            ),
            "excluded_dimensions": ["consistency", "relevance"],
        },
        "raw_audit": raw_audit,
        "expanded_rows": len(records),
        "expert_rating_counts": {
            dimension: [counts[(dimension, label)] for label in LABELS]
            for dimension in DIMENSIONS
        },
        "unanimous_rows": {
            dimension: unanimous[dimension] for dimension in DIMENSIONS
        },
        "split": {
            **split_audit,
            "group_counts": {
                split: len(groups) for split, groups in assignment.items()
            },
            "expanded_row_counts": dict(expanded_counts),
            "split_manifest_sha256": manifest_sha256,
            "checks": split_gates,
        },
        "stage0": {
            "uses_gpu": False,
            "uses_model_predictions": False,
            "test_used_for_model_or_hyperparameter_selection": False,
            "formal_training_authorized": False,
        },
    }


def run(annotation_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = load_annotations(annotation_path)
    raw_audit = audit_raw_annotations(rows)
    records = expand_records(rows)
    assignment, split_audit = choose_group_split(records)
    manifest = serialize_manifest(records, assignment)
    manifest_hash = sha256_bytes(manifest.encode("utf-8"))
    report = build_report(
        annotation_path,
        rows,
        records,
        raw_audit,
        assignment,
        split_audit,
        manifest_hash,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_manifest.jsonl").write_text(manifest, encoding="utf-8")
    (output_dir / "stage0_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SummEval for Exp62")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_ROOT / "stage0")
    args = parser.parse_args()
    report = run(args.annotations, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
