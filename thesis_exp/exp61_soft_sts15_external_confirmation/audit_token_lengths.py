from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import transformers
from transformers import AutoTokenizer

from thesis_exp.exp61_soft_sts15_external_confirmation.audit_dataset import (
    EXPECTED_COMMIT,
    EXPECTED_DATA_SHA256,
    EXPECTED_ROWS,
    OUTPUT_ROOT,
    load_dataset,
    sha256_bytes,
    write_json,
)


EXPECTED_TOKENIZER_FILES = {
    "config.json": "d479c427a9ca5295218063d4f9aca4f297ab4ac27487cca7af42c84643d51ef0",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
}
INPUT_TEMPLATE = (
    "Sentence 1:\n{sentence1}\n\n"
    "Sentence 2:\n{sentence2}\n\n"
    "Task:\nPredict their semantic similarity on a scale from 0 "
    "(completely unrelated) to 5 (semantically equivalent)."
)
MAX_LENGTH_CANDIDATES = (128, 256, 512, 1024)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_manifest(path: Path) -> dict[str, str]:
    manifest = {name: file_sha256(path / name) for name in EXPECTED_TOKENIZER_FILES}
    if manifest != EXPECTED_TOKENIZER_FILES:
        raise RuntimeError(f"tokenizer snapshot mismatch: {manifest}")
    return manifest


def render_input(sentence1: str, sentence2: str) -> str:
    return INPUT_TEMPLATE.format(sentence1=sentence1, sentence2=sentence2)


def choose_max_length(maximum: int) -> int:
    for candidate in MAX_LENGTH_CANDIDATES:
        if maximum <= candidate:
            return candidate
    raise RuntimeError(
        f"maximum tokenized length {maximum} exceeds frozen candidates {MAX_LENGTH_CANDIDATES}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    dataset_path = args.source_repo / "data/text.clean"
    dataset_hash = sha256_bytes(dataset_path.read_bytes())
    if dataset_hash != EXPECTED_DATA_SHA256:
        raise RuntimeError(f"dataset hash mismatch: {dataset_hash}")
    frame = load_dataset(dataset_path, EXPECTED_ROWS)

    stage0_audit_path = args.output_root / "audit/soft_sts15_stage0_audit.json"
    stage0_audit = json.loads(stage0_audit_path.read_text(encoding="utf-8"))
    if stage0_audit["upstream_commit"] != EXPECTED_COMMIT:
        raise RuntimeError("Stage 0 audit upstream commit mismatch")
    manifest_path = args.output_root / "data/split_manifest.jsonl"
    manifest_hash = sha256_bytes(manifest_path.read_bytes())
    if manifest_hash != stage0_audit["split_manifest_sha256"]:
        raise RuntimeError("split manifest differs from revised Stage 0 audit")

    files = tokenizer_manifest(args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    lengths: list[int] = []
    maximum_record: dict[str, Any] | None = None
    for start in range(0, len(frame), 128):
        batch = frame.iloc[start : start + 128]
        texts = [render_input(left, right) for left, right in zip(batch["sentence1"], batch["sentence2"])]
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )
        for row_id, text, length in zip(batch["row_id"], texts, encoded["length"]):
            numeric = int(length)
            lengths.append(numeric)
            if maximum_record is None or numeric > maximum_record["tokens"]:
                maximum_record = {
                    "row_id": int(row_id),
                    "tokens": numeric,
                    "rendered_input_sha256": sha256_bytes(text.encode("utf-8")),
                }

    if len(lengths) != EXPECTED_ROWS or maximum_record is None:
        raise RuntimeError("tokenizer audit did not cover all rows")
    maximum = max(lengths)
    selected = choose_max_length(maximum)
    gates = {
        "all_rows_tokenized": len(lengths) == EXPECTED_ROWS,
        "tokenizer_snapshot_matches_exp60": files == EXPECTED_TOKENIZER_FILES,
        "no_truncation_at_selected_max_length": maximum <= selected,
        "selected_from_frozen_candidates": selected in MAX_LENGTH_CANDIDATES,
        "no_model_weights_loaded": True,
        "no_gpu_used": True,
    }
    if not all(gates.values()):
        raise RuntimeError("tokenizer audit failed closed")

    payload = {
        "status": "EXP61_TOKENIZER_LENGTH_AUDIT_PASS",
        "dataset_sha256": dataset_hash,
        "split_manifest_sha256": manifest_hash,
        "tokenizer_files": files,
        "transformers_version": transformers.__version__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "input_template": INPUT_TEMPLATE,
        "input_template_sha256": sha256_bytes(INPUT_TEMPLATE.encode("utf-8")),
        "add_special_tokens": True,
        "rows": len(lengths),
        "token_lengths": {
            "minimum": min(lengths),
            "median": float(np.quantile(lengths, 0.5)),
            "p90": float(np.quantile(lengths, 0.9)),
            "p95": float(np.quantile(lengths, 0.95)),
            "p99": float(np.quantile(lengths, 0.99)),
            "maximum": maximum,
        },
        "maximum_record": maximum_record,
        "candidate_max_lengths": list(MAX_LENGTH_CANDIDATES),
        "selected_max_length": selected,
        "gates": gates,
        "model_training_performed": False,
        "gpu_used": False,
    }
    write_json(args.output_root / "audit/tokenizer_length_audit.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

