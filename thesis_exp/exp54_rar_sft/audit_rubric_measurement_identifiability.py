"""CPU-only inventory for crossed response--rubric identifiability."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


SCHEMA_VERSION = "exp54-rubric-measurement-identifiability-inventory-v1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def response_id(row: dict[str, Any]) -> str:
    return digest([row["question_key"], row["answer_key"]])


def rubric_text_hash(row: dict[str, Any]) -> str:
    return digest(row["rubric"])


def rubric_id(row: dict[str, Any]) -> str:
    return digest([row["metric_id"], row["language"], rubric_text_hash(row)])


def summary(values: Iterable[int]) -> dict[str, Any]:
    data = sorted(values)
    if not data:
        return {"min": 0, "median": 0, "max": 0, "histogram": {}}
    middle = len(data) // 2
    median = data[middle] if len(data) % 2 else (data[middle - 1] + data[middle]) / 2
    return {"min": data[0], "median": median, "max": data[-1], "histogram": {str(k): v for k, v in sorted(Counter(data).items())}}


def components(adjacency: dict[str, set[str]]) -> list[set[str]]:
    unseen = set(adjacency)
    result = []
    while unseen:
        root = min(unseen)
        seen = {root}
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        unseen -= seen
        result.append(seen)
    return sorted(result, key=lambda group: (-len(group), min(group)))


def articulation_points(adjacency: dict[str, set[str]]) -> set[str]:
    timer = 0
    discovered: dict[str, int] = {}
    low: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    points: set[str] = set()

    def visit(node: str) -> None:
        nonlocal timer
        timer += 1
        discovered[node] = low[node] = timer
        children = 0
        for neighbor in adjacency[node]:
            if neighbor not in discovered:
                parent[neighbor] = node
                children += 1
                visit(neighbor)
                low[node] = min(low[node], low[neighbor])
                if parent[node] is None and children > 1:
                    points.add(node)
                if parent[node] is not None and low[neighbor] >= discovered[node]:
                    points.add(node)
            elif neighbor != parent[node]:
                low[node] = min(low[node], discovered[neighbor])

    for node in sorted(adjacency):
        if node not in discovered:
            parent[node] = None
            visit(node)
    return points


def algebraic_connectivity(nodes: set[str], adjacency: dict[str, set[str]]) -> float:
    if len(nodes) < 2:
        return 0.0
    order = sorted(nodes)
    index = {node: i for i, node in enumerate(order)}
    rows, cols = [], []
    for node in order:
        for neighbor in adjacency[node]:
            if neighbor in index:
                rows.append(index[node]); cols.append(index[neighbor])
    matrix = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(order), len(order)))
    laplacian = sparse.csgraph.laplacian(matrix, normed=False)
    if len(order) <= 3:
        eigenvalues = np.linalg.eigvalsh(laplacian.toarray())
    else:
        eigenvalues = eigsh(laplacian.astype(float), k=2, which="SM", return_eigenvectors=False)
    eigenvalues = sorted(max(0.0, float(value)) for value in eigenvalues)
    # Sparse eigensolvers can vary in their final floating-point tail across
    # equivalent runs. The inventory needs byte-stable public artifacts, while
    # ten decimal places are more than sufficient for this diagnostic.
    return round(eigenvalues[1], 10) if len(eigenvalues) > 1 else 0.0


def mutual_information(pairs: Iterable[tuple[str, str]]) -> float:
    counts = Counter(pairs)
    total = sum(counts.values())
    left = Counter(); right = Counter()
    for (a, b), count in counts.items():
        left[a] += count; right[b] += count
    return sum((count / total) * math.log((count * total) / (left[a] * right[b])) for (a, b), count in counts.items()) if total else 0.0


def cramers_v(pairs: Iterable[tuple[str, str]]) -> float:
    counts = Counter(pairs)
    total = sum(counts.values())
    left_values = sorted({a for a, _ in counts}); right_values = sorted({b for _, b in counts})
    if not total or min(len(left_values), len(right_values)) <= 1:
        return 0.0
    left = Counter(); right = Counter()
    for (a, b), count in counts.items(): left[a] += count; right[b] += count
    chi2 = 0.0
    for a in left_values:
        for b in right_values:
            expected = left[a] * right[b] / total
            chi2 += (counts[(a, b)] - expected) ** 2 / expected
    return math.sqrt((chi2 / total) / min(len(left_values) - 1, len(right_values) - 1))


def anonymized_contingency(rows: list[dict[str, Any]], left_field: str, right_field: str, hash_right: bool = False) -> dict[str, Any]:
    counts = Counter((str(row[left_field]), str(row[right_field])) for row in rows)
    cells = []
    for (left, right), count in sorted(counts.items()):
        cells.append({"left": left, "right": digest(right.encode()) if hash_right else right, "count": count})
    pairs = [(str(row[left_field]), str(row[right_field])) for row in rows]
    return {"mutual_information_nats": mutual_information(pairs), "cramers_v": cramers_v(pairs), "cells": cells}


def build_inventory(train_path: Path, dev_path: Path, output_path: Path) -> dict[str, Any]:
    train = read_jsonl(train_path)
    dev = read_jsonl(dev_path)
    if len(train) != 2654:
        raise ValueError("expected exact 2,654-row RAR train")
    required = {"record_id", "question_key", "answer_key", "metric_id", "metric_canonical", "language", "rubric", "scenario_canonical", "subject_canonical", "label_5", "human_1_5", "human_2_5", "human_3_5"}
    if any(not required <= set(row) for row in train):
        raise ValueError("train row lacks inventory fields")

    edge_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    response_meta: dict[str, dict[str, Any]] = {}
    rubric_meta: dict[str, dict[str, Any]] = {}
    for row in train:
        rid, uid = response_id(row), rubric_id(row)
        edge_rows[(rid, uid)].append(row)
        response_meta.setdefault(rid, {"language": row["language"]})
        rubric_meta.setdefault(uid, {"metric_id": row["metric_id"], "metric": row["metric_canonical"], "language": row["language"], "rubric_sha256": rubric_text_hash(row)})
    duplicate_groups = {edge: rows for edge, rows in edge_rows.items() if len(rows) > 1}
    inconsistent_duplicates = 0
    for rows in duplicate_groups.values():
        signatures = {(row["label_5"], row["human_1_5"], row["human_2_5"], row["human_3_5"]) for row in rows}
        inconsistent_duplicates += len(signatures) > 1

    adjacency: dict[str, set[str]] = {f"R:{key}": set() for key in response_meta} | {f"U:{key}": set() for key in rubric_meta}
    for rid, uid in edge_rows:
        adjacency[f"R:{rid}"].add(f"U:{uid}"); adjacency[f"U:{uid}"].add(f"R:{rid}")
    graph_components = components(adjacency)
    points = articulation_points(adjacency)
    component_reports = []
    for group in graph_components:
        languages = sorted({response_meta[node[2:]]["language"] for node in group if node.startswith("R:")})
        component_reports.append({"node_count": len(group), "response_count": sum(node.startswith("R:") for node in group), "rubric_count": sum(node.startswith("U:") for node in group), "languages": languages, "lambda2": algebraic_connectivity(group, adjacency)})

    response_rubrics: dict[str, set[str]] = defaultdict(set)
    rubric_responses: dict[str, set[str]] = defaultdict(set)
    for rid, uid in edge_rows:
        response_rubrics[rid].add(uid); rubric_responses[uid].add(rid)

    metric_responses: dict[str, set[str]] = defaultdict(set)
    for rid, uid in edge_rows:
        metric_responses[rubric_meta[uid]["metric_id"]].add(rid)
    metric_ids = sorted(metric_responses)
    pair_matrix = [{"metric_a": a, "metric_b": b, "shared_responses": len(metric_responses[a] & metric_responses[b])} for a, b in itertools.combinations(metric_ids, 2)]

    boundary_support = []
    insufficient_nodes = set()
    for uid in sorted(rubric_meta):
        rows = [values[0] for (rid, rubric), values in edge_rows.items() if rubric == uid]
        counts = Counter(int(row["label_5"]) for row in rows)
        boundaries = []
        for k in range(1, 5):
            lower = sum(counts[label] for label in range(1, k + 1)); upper = sum(counts[label] for label in range(k + 1, 6))
            boundaries.append({"k": k, "n_le_k": lower, "n_gt_k": upper, "n_k": counts[k], "n_k_plus_1": counts[k + 1], "both_sides_at_least_20": lower >= 20 and upper >= 20})
            if lower < 20 or upper < 20: insufficient_nodes.add((uid, k))
        boundary_support.append({"rubric_id": uid, **rubric_meta[uid], "label_counts": {str(label): counts[label] for label in range(1, 6)}, "boundaries": boundaries})

    train_responses = set(response_meta)
    dev_response_edges = {(response_id(row), rubric_id(row)) for row in dev}
    dev_responses = {rid for rid, _ in dev_response_edges}
    train_edges = set(edge_rows)
    shared_response_split = train_responses & dev_responses
    cross_metric_exposure = sum(bool(response_rubrics[rid] - {uid for rr, uid in dev_response_edges if rr == rid}) for rid in shared_response_split)

    rubric_texts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in train: rubric_texts[(row["metric_id"], row["language"])].add(rubric_text_hash(row))
    rubric_variation = [{"metric_id": metric, "language": language, "unique_texts": len(texts)} for (metric, language), texts in sorted(rubric_texts.items())]

    confounding = {
        "metric_x_scenario": anonymized_contingency(train, "metric_id", "scenario_canonical"),
        "metric_x_language": anonymized_contingency(train, "metric_id", "language"),
        "metric_x_label": anonymized_contingency(train, "metric_id", "label_5"),
        "metric_x_question": anonymized_contingency(train, "metric_id", "question_key", hash_right=True),
        "scenario_x_label": anonymized_contingency(train, "scenario_canonical", "label_5"),
    }

    pairs_ge_30 = sum(row["shared_responses"] >= 30 for row in pair_matrix)
    all_boundary_supported = not insufficient_nodes
    holdouts = {
        "edge_holdout_constructible": any(len(value) >= 2 for value in response_rubrics.values()),
        "response_holdout_constructible": len(response_meta) >= 5,
        "rubric_holdout_constructible": len(rubric_meta) >= 4,
        "double_holdout_constructible_structurally": len(response_meta) >= 5 and len(rubric_meta) >= 4,
        "current_dev_is_response_disjoint": not shared_response_split,
    }
    gates = {
        "four_metric_pairs_share_at_least_30_responses": pairs_ge_30 >= 4,
        "all_target_rubric_boundaries_have_20_per_side": all_boundary_supported,
        "all_four_holdout_types_structurally_constructible": all(holdouts[key] for key in ("edge_holdout_constructible", "response_holdout_constructible", "rubric_holdout_constructible", "double_holdout_constructible_structurally")),
        "multi_trait_not_scalar_quality_frozen": True,
        "rater_effect_removed_without_verified_identity": True,
        "genuine_external_or_new_rubric_nodes_prepared": False,
        "new_holdout_2_prepared": False,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "decision": "INVENTORY_NO_GO_FOR_CCF_A_COLD_START_OPERATOR" if not all(gates.values()) else "INVENTORY_PASS_TO_LABEL_ONLY_BASELINES",
        "train": {"rows": len(train), "sha256": sha256_file(train_path), "responses": len(response_meta), "rubric_nodes": len(rubric_meta), "unique_edges": len(edge_rows), "duplicate_edge_groups": len(duplicate_groups), "duplicate_edge_rows_beyond_first": sum(len(rows) - 1 for rows in duplicate_groups.values()), "inconsistent_duplicate_groups": inconsistent_duplicates},
        "graph": {"connected_components": len(graph_components), "components": component_reports, "response_degree": summary(len(value) for value in response_rubrics.values()), "rubric_degree": summary(len(value) for value in rubric_responses.values()), "articulation_response_nodes": sum(node.startswith("R:") for node in points), "articulation_rubric_nodes": sum(node.startswith("U:") for node in points)},
        "metric_pair_shared_response_matrix": pair_matrix,
        "metric_pairs_shared_at_least_30": pairs_ge_30,
        "boundary_support": boundary_support,
        "rubric_boundary_pairs_below_20_per_side": len(insufficient_nodes),
        "confounding": confounding,
        "split_exposure": {"dev_rows_metadata_only": len(dev), "train_dev_shared_responses": len(shared_response_split), "train_dev_shared_edges": len(train_edges & dev_response_edges), "shared_responses_with_cross_metric_exposure": cross_metric_exposure},
        "holdouts": holdouts,
        "rubric_variation": rubric_variation,
        "canonical_rubric_texts": sum(len(texts) for texts in rubric_texts.values()),
        "rater_provenance": {"schema_has_persistent_rater_id": False, "source_slots": ["human_1", "human_2", "human_3"], "fixed_rater_severity_authorized": False},
        "gates": gates,
        "test_metadata_accessed_accidentally_during_preformal_discovery": True,
        "test_labels_inspected": False,
        "test_text_inspected": False,
        "formal_inventory_uses_test": False,
        "gpu_used": False,
        "model_calls": 0,
        "hidden_states_extracted": False,
        "training_started": False,
        "privacy": {"question_text_published": False, "answer_text_published": False, "rationale_published": False, "person_name_published": False},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(build_inventory(args.train, args.dev, args.output)["decision"])


if __name__ == "__main__":
    main()
