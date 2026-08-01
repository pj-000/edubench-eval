"""Audit whether train/dev can identify crossed-metric conditional fidelity."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "exp54-crossed-metric-conditional-fidelity-feasibility-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("question_key", "answer_key", "metric_id"):
                if not isinstance(row.get(field), str) or not row[field]:
                    raise ValueError(f"{path}:{line_number}: missing {field}")
            rows.append(row)
    return rows


def crossed_inventory(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    qa_metrics: dict[tuple[str, str], set[str]] = defaultdict(set)
    edge_counts: Counter[tuple[str, str, str]] = Counter()
    row_count = 0
    for row in rows:
        row_count += 1
        qa = (str(row["question_key"]), str(row["answer_key"]))
        metric = str(row["metric_id"])
        qa_metrics[qa].add(metric)
        edge_counts[(qa[0], qa[1], metric)] += 1

    pair_counts: Counter[tuple[str, str]] = Counter()
    metric_count_distribution: Counter[int] = Counter()
    for metrics in qa_metrics.values():
        metric_count_distribution[len(metrics)] += 1
        for pair in itertools.combinations(sorted(metrics), 2):
            pair_counts[pair] += 1

    duplicate_edges = [count for count in edge_counts.values() if count > 1]
    return {
        "row_count": row_count,
        "qa_node_count": len(qa_metrics),
        "multi_metric_qa_node_count": sum(len(metrics) >= 2 for metrics in qa_metrics.values()),
        "single_metric_qa_node_count": sum(len(metrics) == 1 for metrics in qa_metrics.values()),
        "metric_count_distribution": {
            str(key): metric_count_distribution[key] for key in sorted(metric_count_distribution)
        },
        "metric_pair_count": len(pair_counts),
        "maximum_shared_qa_per_metric_pair": max(pair_counts.values(), default=0),
        "duplicate_response_metric_edge_count": len(duplicate_edges),
        "duplicate_extra_row_count": sum(count - 1 for count in duplicate_edges),
        "pair_counts": pair_counts,
    }


def feasibility_report(
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    train = crossed_inventory(train_rows)
    dev = crossed_inventory(dev_rows)
    rules = protocol["go_requirements"]
    train_min = int(rules["minimum_train_shared_qa_per_metric_pair"])
    dev_min = int(rules["minimum_dev_shared_qa_per_metric_pair"])
    min_pairs = int(rules["minimum_jointly_eligible_metric_pairs"])
    min_nodes = int(rules["minimum_multi_metric_dev_qa_nodes"])

    all_pairs = sorted(set(train["pair_counts"]) | set(dev["pair_counts"]))
    eligible_pairs = [
        pair
        for pair in all_pairs
        if train["pair_counts"][pair] >= train_min and dev["pair_counts"][pair] >= dev_min
    ]
    top_dev_pairs = sorted(
        (
            {
                "metric_a": pair[0],
                "metric_b": pair[1],
                "train_shared_qa": int(train["pair_counts"][pair]),
                "dev_shared_qa": int(dev["pair_counts"][pair]),
            }
            for pair in dev["pair_counts"]
        ),
        key=lambda row: (-row["dev_shared_qa"], row["metric_a"], row["metric_b"]),
    )[:20]

    node_gate = dev["multi_metric_qa_node_count"] >= min_nodes
    pair_gate = len(eligible_pairs) >= min_pairs
    decision = "GO_FULL_CONDITIONAL_FIDELITY_AUDIT" if node_gate and pair_gate else "NO_GO_INSUFFICIENT_CROSSED_METRIC_SUPPORT"

    def public_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in inventory.items() if key != "pair_counts"}

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "data_scope": {
            "splits": ["train", "dev"],
            "new_training": False,
            "new_inference": False,
            "new_api_calls": False,
            "gpu_used": False,
            "test_accessed": False,
        },
        "inventory": {"train": public_inventory(train), "dev": public_inventory(dev)},
        "thresholds": {
            "minimum_multi_metric_dev_qa_nodes": min_nodes,
            "minimum_train_shared_qa_per_metric_pair": train_min,
            "minimum_dev_shared_qa_per_metric_pair": dev_min,
            "minimum_jointly_eligible_metric_pairs": min_pairs,
        },
        "gate_results": {
            "multi_metric_dev_qa_nodes": {
                "observed": dev["multi_metric_qa_node_count"],
                "passed": node_gate,
            },
            "jointly_eligible_metric_pairs": {
                "observed": len(eligible_pairs),
                "passed": pair_gate,
            },
            "train_metric_pairs_with_at_least_30_shared_qa": sum(
                count >= train_min for count in train["pair_counts"].values()
            ),
            "dev_metric_pairs_with_at_least_20_shared_qa": sum(
                count >= dev_min for count in dev["pair_counts"].values()
            ),
        },
        "jointly_eligible_metric_pairs": [
            {
                "metric_a": pair[0],
                "metric_b": pair[1],
                "train_shared_qa": int(train["pair_counts"][pair]),
                "dev_shared_qa": int(dev["pair_counts"][pair]),
            }
            for pair in eligible_pairs
        ],
        "top_dev_metric_pairs": top_dev_pairs,
        "downstream_estimands": {
            "human_metric_difference_prevalence": "NOT_ESTIMATED_PAIR_SUPPORT_GATE_FAILED" if not pair_gate else "AUTHORIZED",
            "leave_one_rater_metric_concordance": "NOT_ESTIMATED_PAIR_SUPPORT_GATE_FAILED" if not pair_gate else "AUTHORIZED",
            "model_conditional_fidelity": "NOT_ESTIMATED_PAIR_SUPPORT_GATE_FAILED" if not pair_gate else "AUTHORIZED",
        },
        "interpretation": {
            "supported": "The current dev split does not meet the preregistered repeated-QA metric-pair support required for a stable conditional-fidelity direction-selection audit." if not pair_gate else "The current split meets the inventory requirements for the full conditional-fidelity audit.",
            "not_supported": "A NO-GO does not show that rubric conditional insensitivity is absent; it shows that the current dev split cannot identify it under the specified design.",
            "new_method_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--test-source", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("protocol schema differs")
    report = feasibility_report(read_jsonl(args.train), read_jsonl(args.dev), protocol)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lock = {
        "schema_version": SCHEMA_VERSION,
        "decision": report["decision"],
        "train_sha256": sha256_file(args.train),
        "dev_sha256": sha256_file(args.dev),
        "protocol_sha256": sha256_file(args.protocol),
        "analysis_source_sha256": sha256_file(args.source),
        "test_source_sha256": sha256_file(args.test_source),
        "public_report_sha256": sha256_file(args.report),
        "new_training": False,
        "new_inference": False,
        "new_api_calls": False,
        "gpu_used": False,
        "test_accessed": False,
    }
    args.lock.write_text(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["decision"])


if __name__ == "__main__":
    main()
