"""Placeholder API interface for Exp6-2 generation.

This module intentionally does not call any API. It documents where a future
DeepSeek V4 Pro integration would live after manual approval.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from thesis_exp.src.edujudge.exp06_generation import DEFAULT_GENERATION_MODEL


def generate_with_api(*_: object, **__: object) -> None:
    raise NotImplementedError(
        "Exp6-2 scaffold does not call APIs. Implement this only after manual approval, "
        "with train-only prompts and leakage checks."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print planned API interface without calling any model")
    parser.add_argument("--input-prompts", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL)
    args = parser.parse_args()
    if args.dry_run:
        print(
            "DRY RUN ONLY: would generate from "
            f"{args.input_prompts or '<prompts jsonl>'} using model={args.model} "
            f"and write {args.output_jsonl or '<synthetic jsonl>'}."
        )
        return
    generate_with_api(args)


if __name__ == "__main__":
    main()
