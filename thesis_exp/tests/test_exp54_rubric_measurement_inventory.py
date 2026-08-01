from __future__ import annotations

import pytest

from thesis_exp.exp54_rar_sft.audit_rubric_measurement_identifiability import (
    articulation_points,
    build_inventory,
    components,
    cramers_v,
    mutual_information,
    response_id,
    rubric_id,
    summary,
)

from thesis_exp.exp54_rar_sft import REPO_ROOT


def test_ids_are_structural_and_deterministic() -> None:
    row = {"question_key": "q", "answer_key": "a", "metric_id": "m", "language": "en", "rubric": ["r"]}
    assert response_id(row) == response_id(dict(row))
    assert rubric_id(row) == rubric_id(dict(row))
    changed = dict(row, language="zh")
    assert rubric_id(row) != rubric_id(changed)


def test_components_and_articulation_points() -> None:
    graph = {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}, "d": set()}
    assert [len(group) for group in components(graph)] == [3, 1]
    assert articulation_points(graph) == {"b"}


def test_summary_even_median_and_histogram() -> None:
    result = summary([1, 2, 2, 5])
    assert result["median"] == 2
    assert result["histogram"] == {"1": 1, "2": 2, "5": 1}


def test_independence_statistics() -> None:
    independent = [(a, b) for a in "ab" for b in "xy" for _ in range(2)]
    assert mutual_information(independent) == pytest.approx(0)
    assert cramers_v(independent) == pytest.approx(0)
    associated = [("a", "x")] * 5 + [("b", "y")] * 5
    assert mutual_information(associated) > 0
    assert cramers_v(associated) == pytest.approx(1)


def test_locked_rar_train_inventory_regression(tmp_path) -> None:
    report = build_inventory(
        REPO_ROOT / "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl",
        REPO_ROOT / "thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl",
        tmp_path / "report.json",
    )
    assert report["train"] == {
        "rows": 2654,
        "sha256": "0a1733b9209984c5c4291d205d1ac6057bed341717903b9de075d07de44a878e",
        "responses": 906,
        "rubric_nodes": 24,
        "unique_edges": 2637,
        "duplicate_edge_groups": 16,
        "duplicate_edge_rows_beyond_first": 17,
        "inconsistent_duplicate_groups": 1,
    }
    assert report["graph"]["connected_components"] == 2
    assert report["metric_pairs_shared_at_least_30"] == 41
    assert report["rubric_boundary_pairs_below_20_per_side"] == 68
    assert report["split_exposure"]["train_dev_shared_responses"] == 472
    assert report["decision"] == "INVENTORY_NO_GO_FOR_CCF_A_COLD_START_OPERATOR"
