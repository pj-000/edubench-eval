"""Tokenizer-bound no-truncation audit for all frozen Exp62 inputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.exp62_summeval_routing_confirmation import OUTPUT_ROOT
from thesis_exp.exp62_summeval_routing_confirmation.audit_dataset import (
    expand_records,
    load_annotations,
    sha256_file,
)
from thesis_exp.exp62_summeval_routing_confirmation.data import render_input


FROZEN_MAX_LENGTH = 256


def run(annotation_path: Path, tokenizer_path: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=False
    )
    records = expand_records(load_annotations(annotation_path))
    lengths: list[int] = []
    by_dimension: dict[str, list[int]] = {"coherence": [], "fluency": []}
    for item in records:
        text = render_input(item["summary"], item["dimension"])
        length = len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        lengths.append(length)
        by_dimension[item["dimension"]].append(length)
    maximum = max(lengths)
    checks = {
        "all_3200_inputs_audited": len(lengths) == 3200,
        "maximum_within_frozen_limit": maximum <= FROZEN_MAX_LENGTH,
        "truncation_not_required": maximum <= FROZEN_MAX_LENGTH,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Exp62 token-length audit failed: {checks}; maximum={maximum}")
    ordered = sorted(lengths)
    return {
        "status": "EXP62_TOKEN_LENGTH_AUDIT_PASS",
        "annotation_sha256": sha256_file(annotation_path),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "tokenizer_json_sha256": sha256_file(tokenizer_path / "tokenizer.json"),
        "records": len(lengths),
        "maximum_tokens": maximum,
        "p95_tokens": ordered[int(0.95 * len(ordered)) - 1],
        "p99_tokens": ordered[int(0.99 * len(ordered)) - 1],
        "maximum_by_dimension": {
            dimension: max(values) for dimension, values in by_dimension.items()
        },
        "frozen_max_length": FROZEN_MAX_LENGTH,
        "length_histogram": dict(sorted(Counter(lengths).items())),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--tokenizer_path", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "stage0/token_length_audit.json",
    )
    args = parser.parse_args()
    result = run(args.annotations, args.tokenizer_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

