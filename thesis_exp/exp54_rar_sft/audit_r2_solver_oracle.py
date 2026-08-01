"""Audit the R2 strict-permutation matcher against a brute-force oracle.

This audit is tokenizer-independent. It proves on fixed adversarial cases and
deterministic random strata that the optimized matcher uses the lexicographic
objective required by the preregistration:

1. maximize the active strict-permutation subset;
2. among maximum-coverage solutions, minimize total length difference.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import file_sha256, write_json
from thesis_exp.exp54_rar_sft.build_r2_donor_map import match_stratum


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
    "r2_solver_oracle_report.json"
)
DEFAULT_SEED = 20260723
DEFAULT_RANDOM_CASES = 64
MAX_ORACLE_SIZE = 8


def synthetic_item(
    sample_id: str,
    reference_id: str,
    length: int,
    *,
    qa_key: str | None = None,
) -> dict[str, Any]:
    if length < 1:
        raise ValueError("synthetic rationale length must be positive")
    return {
        "record_id": sample_id,
        "normalized_qa_key": qa_key or f"qa-{sample_id}",
        "reference_id": reference_id,
        "reference_index": 0,
        "label_5": 2,
        "metric_id": "IFTC",
        "language": "zh",
        "reason": "x" * length,
        "reason_sha256": reference_id,
    }


def brute_force_objective(items: list[dict[str, Any]]) -> tuple[int, int]:
    """Return maximum active count and conditional minimum length cost."""
    if len(items) > MAX_ORACLE_SIZE:
        raise ValueError(f"oracle supports at most {MAX_ORACLE_SIZE} references")
    ordered = sorted(items, key=lambda item: item["reference_id"])
    lengths = [len(item["reason"]) for item in ordered]
    best_active = -1
    best_length_cost = 10**18
    for active_count in range(len(ordered) + 1):
        for active_indices in itertools.combinations(range(len(ordered)), active_count):
            for donor_indices in itertools.permutations(active_indices):
                legal = all(
                    recipient_index != donor_index
                    and ordered[recipient_index]["record_id"]
                    != ordered[donor_index]["record_id"]
                    and ordered[recipient_index]["normalized_qa_key"]
                    != ordered[donor_index]["normalized_qa_key"]
                    for recipient_index, donor_index in zip(
                        active_indices,
                        donor_indices,
                    )
                )
                if not legal:
                    continue
                length_cost = sum(
                    abs(lengths[recipient_index] - lengths[donor_index])
                    for recipient_index, donor_index in zip(
                        active_indices,
                        donor_indices,
                    )
                )
                if active_count > best_active:
                    best_active = active_count
                    best_length_cost = length_cost
                elif active_count == best_active:
                    best_length_cost = min(best_length_cost, length_cost)
    return best_active, best_length_cost


def mapping_objective(mapping: list[dict[str, Any]]) -> tuple[int, int]:
    active = [row for row in mapping if row["active"]]
    return len(active), sum(
        int(row["absolute_length_difference"]) for row in active
    )


def canonical_mapping(mapping: list[dict[str, Any]]) -> str:
    payload = [
        {
            "recipient": row["recipient_reference_id"],
            "donor": row["donor_reference_id"],
            "active": row["active"],
            "cycle_id": row["cycle_id"],
        }
        for row in sorted(mapping, key=lambda row: row["recipient_reference_id"])
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def adversarial_cases() -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        (
            "singleton_sample_multiple_references",
            [
                synthetic_item("a", "a:1", 1),
                synthetic_item("a", "a:2", 30),
            ],
        ),
        (
            "two_sample_full_derangement",
            [
                synthetic_item("a", "a:1", 1),
                synthetic_item("b", "b:1", 50),
            ],
        ),
        (
            "unbalanced_three_to_two",
            [
                synthetic_item("a", "a:1", 1),
                synthetic_item("a", "a:2", 2),
                synthetic_item("a", "a:3", 3),
                synthetic_item("b", "b:1", 100),
                synthetic_item("b", "b:2", 101),
            ],
        ),
        (
            "same_content_different_ids_forbidden",
            [
                synthetic_item("a", "a:1", 1, qa_key="duplicate"),
                synthetic_item("b", "b:1", 2, qa_key="duplicate"),
            ],
        ),
        (
            "coverage_must_dominate_length",
            [
                synthetic_item("a", "a:1", 1),
                synthetic_item("b", "b:1", 2),
                synthetic_item("c", "c:1", 10000),
            ],
        ),
        (
            "eight_reference_mixed_constraints",
            [
                synthetic_item("a", "a:1", 1),
                synthetic_item("a", "a:2", 2),
                synthetic_item("b", "b:1", 3),
                synthetic_item("c", "c:1", 5, qa_key="shared-1"),
                synthetic_item("d", "d:1", 8, qa_key="shared-1"),
                synthetic_item("e", "e:1", 13),
                synthetic_item("f", "f:1", 21),
                synthetic_item("g", "g:1", 34),
            ],
        ),
    ]


def random_case(randomizer: random.Random, case_index: int) -> list[dict[str, Any]]:
    size = randomizer.randint(1, 7)
    output = []
    for reference_index in range(size):
        sample_index = randomizer.randint(0, max(0, size // 2))
        qa_index = randomizer.randint(0, max(0, size // 3))
        output.append(
            synthetic_item(
                f"sample-{sample_index}",
                f"random-{case_index}:ref-{reference_index}",
                randomizer.randint(1, 80),
                qa_key=f"qa-{qa_index}",
            )
        )
    return output


def run_audit(seed: int, random_cases: int) -> dict[str, Any]:
    randomizer = random.Random(seed)
    cases = adversarial_cases() + [
        (f"random_{index:03d}", random_case(randomizer, index))
        for index in range(random_cases)
    ]
    failures: list[dict[str, Any]] = []
    order_failures: list[str] = []
    for name, items in cases:
        expected = brute_force_objective(items)
        mapping = match_stratum(items, len)
        actual = mapping_objective(mapping)
        if actual != expected:
            failures.append(
                {
                    "case": name,
                    "expected": list(expected),
                    "actual": list(actual),
                }
            )
        expected_mapping = canonical_mapping(mapping)
        for trial in range(20):
            shuffled = list(items)
            random.Random(f"{seed}:{name}:{trial}").shuffle(shuffled)
            if canonical_mapping(match_stratum(shuffled, len)) != expected_mapping:
                order_failures.append(f"{name}:{trial}")
                break

    checks = {
        "maximum_coverage_matches_brute_force": not failures,
        "conditional_minimum_length_matches_brute_force": not failures,
        "input_order_invariance_20_shuffles": not order_failures,
        "adversarial_cases_included": len(adversarial_cases()) == 6,
        "random_cases_exact": len(cases) - len(adversarial_cases()) == random_cases,
    }
    return {
        "status": "R2_MATCHER_ORACLE_PASS" if all(checks.values()) else "R2_MATCHER_ORACLE_FAIL",
        "seed": seed,
        "max_oracle_size": MAX_ORACLE_SIZE,
        "adversarial_cases": len(adversarial_cases()),
        "random_cases": random_cases,
        "total_cases": len(cases),
        "input_order_trials_per_case": 20,
        "checks": checks,
        "objective_failures": failures,
        "input_order_failures": order_failures,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "solver": "scipy.optimize.linear_sum_assignment",
        },
        "matcher_source_sha256": file_sha256(
            REPO_ROOT / "thesis_exp/exp54_rar_sft/build_r2_donor_map.py"
        ),
        "audit_source_sha256": file_sha256(Path(__file__)),
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--random-cases", type=int, default=DEFAULT_RANDOM_CASES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_audit(args.seed, args.random_cases)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "R2_MATCHER_ORACLE_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
