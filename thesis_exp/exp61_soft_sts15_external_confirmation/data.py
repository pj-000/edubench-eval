"""Fail-closed Soft-STS-15 loader for Exp61 training and development only."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.exp61_soft_sts15_external_confirmation import (
    SPLIT_MANIFEST,
    TRAINING_SPLITS,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.audit_dataset import (
    EXPECTED_COMMIT,
    EXPECTED_DATA_SHA256,
    EXPECTED_ROWS,
    EXPECTED_SPLIT_ROWS,
    empirical_distribution,
    load_dataset,
    normalized_sentence,
    quantized_mean_target,
    sha256_bytes,
    stable_json,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.method import target_fifths


INPUT_TEMPLATE = (
    "Sentence 1:\n{sentence1}\n\n"
    "Sentence 2:\n{sentence2}\n\n"
    "Task:\nPredict their semantic similarity on a scale from 0 "
    "(completely unrelated) to 5 (semantically equivalent)."
)
EXPECTED_MANIFEST_SHA256 = "73b04b02c3aaac303e8ba3b9213030a9dcdf1e5736f18bf3f81a987be8da56ad"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(source_repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_repo, text=True
    ).strip()


def _target_hash(scores: tuple[int, ...]) -> str:
    # This is the exact Stage-0 manifest formula. The derived distribution,
    # mean and hard target are checked separately below.
    return sha256_bytes(stable_json(list(scores)).encode("utf-8"))


def read_manifest(path: Path = SPLIT_MANIFEST) -> dict[int, dict[str, Any]]:
    if _sha256(path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Exp61 split manifest hash mismatch")
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        row_id = int(item["row_id"])
        if row_id in rows:
            raise RuntimeError(f"duplicate manifest row_id: {row_id}")
        rows[row_id] = item
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError("Exp61 manifest row count mismatch")
    return rows


def verify_source(source_repo: Path) -> Path:
    source_repo = source_repo.resolve()
    if _git_commit(source_repo) != EXPECTED_COMMIT:
        raise RuntimeError("Soft-STS-15 source commit mismatch")
    data_path = source_repo / "data/text.clean"
    if _sha256(data_path) != EXPECTED_DATA_SHA256:
        raise RuntimeError("Soft-STS-15 data hash mismatch")
    return data_path


def load_model_rows(source_repo: Path, split: str) -> list[dict[str, Any]]:
    """Load only train/dev; this function has no test override by design."""

    if split not in TRAINING_SPLITS:
        raise PermissionError("Exp61 training loader permits only train and dev")
    frame = load_dataset(verify_source(source_repo), expected_rows=EXPECTED_ROWS)
    manifest = read_manifest()
    rows: list[dict[str, Any]] = []
    for raw in frame.itertuples(index=False):
        item = manifest[int(raw.row_id)]
        if item["split"] != split:
            continue
        scores = tuple(int(value) for value in raw.scores)
        if _target_hash(scores) != item["target_sha256"]:
            raise RuntimeError(f"target hash mismatch for row {raw.row_id}")
        label = quantized_mean_target(scores)
        if label != int(item["hard_label"]):
            raise RuntimeError(f"hard-label mismatch for row {raw.row_id}")
        pair_payload = "\n".join(
            (normalized_sentence(raw.sentence1), normalized_sentence(raw.sentence2))
        ).encode("utf-8")
        if sha256_bytes(pair_payload) != item["pair_sha256"]:
            raise RuntimeError(f"pair hash mismatch for row {raw.row_id}")
        rows.append(
            {
                "record_id": f"softsts15:{int(raw.row_id)}",
                "row_id": int(raw.row_id),
                "text": INPUT_TEMPLATE.format(
                    sentence1=raw.sentence1, sentence2=raw.sentence2
                ),
                "label": label,
                "human_mean": float(sum(scores) / len(scores)),
                "published_scores": list(scores),
                "target_fifths": list(target_fifths(scores)),
                "soft_target": empirical_distribution(scores),
                "component_sha256": item["component_sha256"],
                "pair_sha256": item["pair_sha256"],
                "origin": raw.origin,
                "split": split,
            }
        )
    if len(rows) != EXPECTED_SPLIT_ROWS[split]:
        raise RuntimeError(f"Exp61 {split} row count mismatch")
    return rows


def rows_contract_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value["record_id"]):
        digest.update(
            (
                f"{row['record_id']}\t{row['label']}\t{row['target_fifths']}\t"
                f"{row['component_sha256']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()
