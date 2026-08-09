"""Fail-closed train/dev loader for the frozen Exp62 SummEval split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp62_summeval_routing_confirmation import DIMENSIONS, SPLIT_MANIFEST
from thesis_exp.exp62_summeval_routing_confirmation.audit_dataset import (
    EXPECTED_ANNOTATION_SHA256,
    EXPECTED_EXPANDED_ROWS,
    EXPECTED_RAW_ROWS,
    empirical_target,
    expand_records,
    load_annotations,
    normalized_text,
    quantized_mean_label,
    sha256_bytes,
    sha256_file,
    stable_json,
)


EXPECTED_MANIFEST_SHA256 = "b32a6a78242f7959fe3074e9df817a5e6f87bf447a6f84dc326370311b80fff9"
TRAINING_SPLITS = ("train", "dev")
DIMENSION_RUBRICS = {
    "coherence": (
        "Evaluate the summary as a whole: whether it is well structured, well organized, "
        "and presents information in a coherent order."
    ),
    "fluency": (
        "Evaluate the quality of the individual sentences: grammar, spelling, word choice, "
        "and sentence readability."
    ),
}
INPUT_TEMPLATE = (
    "Candidate summary:\n{summary}\n\n"
    "Evaluation dimension: {dimension}\n"
    "Criterion: {rubric}\n\n"
    "Task: predict the expert rating from 1 (very poor) to 5 (very good)."
)


def render_input(summary: str, dimension: str) -> str:
    if dimension not in DIMENSIONS:
        raise ValueError(f"unsupported Exp62 dimension: {dimension}")
    return INPUT_TEMPLATE.format(
        summary=summary,
        dimension=dimension,
        rubric=DIMENSION_RUBRICS[dimension],
    )


def read_manifest(path: Path = SPLIT_MANIFEST) -> dict[str, dict[str, Any]]:
    if sha256_file(path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Exp62 split manifest hash mismatch")
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        record_id = str(item["record_id"])
        if record_id in rows:
            raise RuntimeError(f"duplicate Exp62 manifest record: {record_id}")
        rows[record_id] = item
    if len(rows) != sum(EXPECTED_EXPANDED_ROWS.values()):
        raise RuntimeError("Exp62 manifest row count mismatch")
    return rows


def _load_frozen_rows(annotation_path: Path, split: str) -> list[dict[str, Any]]:
    if split not in EXPECTED_EXPANDED_ROWS:
        raise ValueError(f"unknown Exp62 split: {split}")
    if sha256_file(annotation_path) != EXPECTED_ANNOTATION_SHA256:
        raise RuntimeError("Exp62 annotation hash mismatch")
    raw = load_annotations(annotation_path)
    expanded = expand_records(raw)
    manifest = read_manifest()
    rows: list[dict[str, Any]] = []
    for item in expanded:
        frozen = manifest[item["record_id"]]
        checks = {
            "group_id": frozen["group_id"] == item["group_id"],
            "model_id": frozen["model_id"] == item["model_id"],
            "dimension": frozen["dimension"] == item["dimension"],
            "summary_sha256": frozen["summary_sha256"] == item["summary_sha256"],
            "target_sha256": frozen["target_sha256"] == item["target_sha256"],
            "hard_label": int(frozen["hard_label"]) == item["hard_label"],
        }
        if not all(checks.values()):
            raise RuntimeError(f"Exp62 frozen row mismatch {item['record_id']}: {checks}")
        if frozen["split"] != split:
            continue
        rows.append(
            {
                "record_id": item["record_id"],
                "group_id": item["group_id"],
                "model_id": item["model_id"],
                "dimension": item["dimension"],
                "text": render_input(item["summary"], item["dimension"]),
                "label": int(item["hard_label"]) - 1,
                "hard_label": int(item["hard_label"]),
                "human_mean": float(item["human_mean"]),
                "expert_scores": list(item["expert_scores"]),
                "soft_target": list(item["soft_target"]),
                "split": split,
            }
        )
    if len(rows) != EXPECTED_EXPANDED_ROWS[split]:
        raise RuntimeError(f"Exp62 {split} row count mismatch")
    return rows


def load_model_rows(annotation_path: Path, split: str) -> list[dict[str, Any]]:
    if split not in TRAINING_SPLITS:
        raise PermissionError("Exp62 training loader permits only train and dev")
    return _load_frozen_rows(annotation_path, split)


def load_test_rows_once(annotation_path: Path) -> list[dict[str, Any]]:
    """Private one-shot loader; formal training must never import this symbol."""

    return _load_frozen_rows(annotation_path, "test")


def rows_contract_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value["record_id"]):
        digest.update(
            stable_json(
                {
                    "record_id": row["record_id"],
                    "group_id": row["group_id"],
                    "dimension": row["dimension"],
                    "label": row["label"],
                    "expert_scores": row["expert_scores"],
                }
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()

