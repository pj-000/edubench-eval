#!/usr/bin/env python3
"""Audit whether the rater mean uniquely determines the empirical distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SPLITS = ("train", "dev", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def distribution(row: dict[str, Any]) -> tuple[int, ...]:
    counts = [0] * 5
    for index in (1, 2, 3):
        score = int(round(float(row[f"human_{index}_5"])))
        counts[score - 1] += 1
    return tuple(counts)


def audit_split(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    distributions_by_mean: dict[str, set[tuple[int, ...]]] = {}
    means_by_distribution: dict[tuple[int, ...], set[str]] = {}
    for row in rows:
        mean_key = f"{float(row['human_mean_5']):.8f}"
        target = distribution(row)
        distributions_by_mean.setdefault(mean_key, set()).add(target)
        means_by_distribution.setdefault(target, set()).add(mean_key)
    ambiguous_means = {
        mean: sorted(list(values))
        for mean, values in distributions_by_mean.items()
        if len(values) > 1
    }
    ambiguous_distributions = {
        str(target): sorted(values)
        for target, values in means_by_distribution.items()
        if len(values) > 1
    }
    return {
        "rows": len(rows),
        "unique_mean_values": len(distributions_by_mean),
        "unique_empirical_distributions": len(means_by_distribution),
        "mean_uniquely_determines_distribution": not ambiguous_means,
        "distribution_uniquely_determines_mean": not ambiguous_distributions,
        "ambiguous_means": ambiguous_means,
        "ambiguous_distributions": ambiguous_distributions,
        "mean_values": sorted(float(value) for value in distributions_by_mean),
        "split_sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "status": "BIJECTIVE_WITHIN_EACH_OBSERVED_SPLIT",
        "interpretation": (
            "Within the observed rating patterns, the continuous three-rater "
            "mean and the five-class empirical count distribution are "
            "one-to-one. Comparisons can identify target geometry/loss effects, "
            "not additional label information."
        ),
        "splits": {
            split: audit_split(args.split_root / f"{split}.jsonl")
            for split in SPLITS
        },
    }
    if not all(
        value["mean_uniquely_determines_distribution"]
        and value["distribution_uniquely_determines_mean"]
        for value in report["splits"].values()
    ):
        report["status"] = "NOT_BIJECTIVE"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
