"""Independently audit the event-level matcher against exhaustive search."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import random
from pathlib import Path
from typing import Any

import scipy

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import file_sha256, write_json
from thesis_exp.exp54_rar_sft.build_r2_event_donor_map import (
    CORE_MATCHER_SOURCE,
    match_event_stratum,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
    "r2_event_solver_oracle_report.json"
)
EVENT_MATCHER_SOURCE = (
    REPO_ROOT / "thesis_exp/exp54_rar_sft/build_r2_event_donor_map.py"
)
DEFAULT_SEED = 20260723
DEFAULT_RANDOM_CASES = 64
MAX_ORACLE_SIZE = 8


def synthetic_event(
    record_id: str,
    event_id: str,
    length: int,
    *,
    qa_key: str | None = None,
    reference_id: str | None = None,
) -> dict[str, Any]:
    if length < 1:
        raise ValueError("synthetic token length must be positive")
    reason = "x" * length
    return {
        "event_id": event_id,
        "seed": 42,
        "epoch_index": 0,
        "epoch_number": 1,
        "row_position": 0,
        "record_id": record_id,
        "normalized_qa_key": qa_key or f"qa-{record_id}",
        "reference_id": reference_id or f"{record_id}:human_1",
        "label_5": 2,
        "metric_id": "IFTC",
        "language": "zh",
        "reason": reason,
        "reason_bytes_sha256": f"bytes-{event_id}",
        "rationale_token_length": length,
        "rationale_token_ids_sha256": hashlib.sha256(
            event_id.encode("utf-8")
        ).hexdigest(),
    }


def brute_force_objective(items: list[dict[str, Any]]) -> tuple[int, int]:
    """Return maximum active events and conditional minimum length cost."""
    if len(items) > MAX_ORACLE_SIZE:
        raise ValueError(f"Oracle supports at most {MAX_ORACLE_SIZE} events")
    ordered = sorted(items, key=lambda item: item["event_id"])
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
                    abs(
                        int(ordered[recipient_index]["rationale_token_length"])
                        - int(ordered[donor_index]["rationale_token_length"])
                    )
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
        int(row["absolute_token_length_difference"]) for row in active
    )


def canonical_mapping(mapping: list[dict[str, Any]]) -> str:
    payload = [
        {
            "recipient": row["recipient_event_id"],
            "donor": row["donor_event_id"],
            "active": row["active"],
            "cycle_id": row["cycle_id"],
        }
        for row in sorted(mapping, key=lambda row: row["recipient_event_id"])
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def adversarial_cases() -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        (
            "singleton",
            [synthetic_event("a", "event-a", 1)],
        ),
        (
            "two_rows_full_derangement",
            [
                synthetic_event("a", "event-a", 1),
                synthetic_event("b", "event-b", 50),
            ],
        ),
        (
            "same_record_forbidden",
            [
                synthetic_event("a", "event-a1", 1),
                synthetic_event("a", "event-a2", 2),
                synthetic_event("b", "event-b", 3),
            ],
        ),
        (
            "same_content_different_rows_forbidden",
            [
                synthetic_event("a", "event-a", 1, qa_key="duplicate"),
                synthetic_event("b", "event-b", 2, qa_key="duplicate"),
            ],
        ),
        (
            "coverage_dominates_length",
            [
                synthetic_event("a", "event-a", 1),
                synthetic_event("b", "event-b", 2),
                synthetic_event("c", "event-c", 10000),
            ],
        ),
        (
            "same_reference_identity_distinct_occurrences",
            [
                synthetic_event(
                    "a",
                    "event-a-occurrence-1",
                    4,
                    reference_id="shared-reference",
                ),
                synthetic_event(
                    "b",
                    "event-b",
                    5,
                    reference_id="other-reference",
                ),
            ],
        ),
        (
            "eight_event_mixed_constraints",
            [
                synthetic_event("a", "event-a1", 1),
                synthetic_event("a", "event-a2", 2),
                synthetic_event("b", "event-b", 3),
                synthetic_event("c", "event-c", 5, qa_key="shared"),
                synthetic_event("d", "event-d", 8, qa_key="shared"),
                synthetic_event("e", "event-e", 13),
                synthetic_event("f", "event-f", 21),
                synthetic_event("g", "event-g", 34),
            ],
        ),
    ]


def random_case(randomizer: random.Random, case_index: int) -> list[dict[str, Any]]:
    size = randomizer.randint(1, 7)
    return [
        synthetic_event(
            f"sample-{randomizer.randint(0, max(0, size // 2))}",
            f"random-{case_index:03d}-event-{event_index}",
            randomizer.randint(1, 80),
            qa_key=f"qa-{randomizer.randint(0, max(0, size // 3))}",
        )
        for event_index in range(size)
    ]


def run_audit(seed: int, random_cases: int) -> dict[str, Any]:
    randomizer = random.Random(seed)
    cases = adversarial_cases() + [
        (f"random_{index:03d}", random_case(randomizer, index))
        for index in range(random_cases)
    ]
    objective_failures = []
    order_failures = []
    for name, items in cases:
        expected = brute_force_objective(items)
        mapping = match_event_stratum(items)
        actual = mapping_objective(mapping)
        if actual != expected:
            objective_failures.append(
                {"case": name, "expected": list(expected), "actual": list(actual)}
            )
        canonical = canonical_mapping(mapping)
        for trial in range(20):
            shuffled = list(items)
            random.Random(f"{seed}:{name}:{trial}").shuffle(shuffled)
            if canonical_mapping(match_event_stratum(shuffled)) != canonical:
                order_failures.append(f"{name}:{trial}")
    status = (
        "R2_EVENT_MATCHER_ORACLE_PASS"
        if not objective_failures and not order_failures
        else "R2_EVENT_MATCHER_ORACLE_FAIL"
    )
    return {
        "status": status,
        "oracle_independent_of_production_solver": True,
        "oracle_method": "exhaustive active-subset and permutation enumeration",
        "seed": seed,
        "fixed_adversarial_cases": len(adversarial_cases()),
        "random_cases": random_cases,
        "total_cases": len(cases),
        "maximum_oracle_size": MAX_ORACLE_SIZE,
        "input_order_shuffles_per_case": 20,
        "objective_failures": objective_failures,
        "input_order_failures": order_failures,
        "event_matcher_source_sha256": file_sha256(EVENT_MATCHER_SOURCE),
        "core_matcher_source_sha256": file_sha256(CORE_MATCHER_SOURCE),
        "runtime": {
            "python": platform.python_version(),
            "scipy": scipy.__version__,
        },
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
        "formal_training_allowed": False,
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
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "R2_EVENT_MATCHER_ORACLE_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
