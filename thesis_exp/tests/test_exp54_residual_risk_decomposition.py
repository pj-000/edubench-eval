from __future__ import annotations

import math

import numpy as np
import pytest

from thesis_exp.exp54_rar_sft.audit_residual_risk_decomposition import audit
from thesis_exp.exp54_rar_sft.analyze_residual_risk_decomposition import (
    _ambiguity,
    _prediction_map,
    annotate_dev,
    boundary_cell_support_association,
    clustered_gap_bootstrap,
    holm_adjust,
    landing_report,
    leave_one_group_out_gaps,
    score_metrics,
)


def _row(record_id: str, *, label: int, metric: str = "m", language: str = "en", answer: str | None = None, raters: tuple[int, int, int] | None = None) -> dict:
    values = raters or (label, label, label)
    return {
        "record_id": record_id,
        "question_key": f"q-{record_id.split('-')[0]}",
        "answer_key": answer or f"a-{record_id}",
        "metric_id": metric,
        "language": language,
        "scenario_canonical": "single",
        "label_5": label,
        "human_1_5": values[0],
        "human_2_5": values[1],
        "human_3_5": values[2],
    }


def test_annotations_separate_response_edge_support_and_ambiguity() -> None:
    train = []
    for label in range(1, 6):
        for index in range(25):
            train.append(_row(f"t{label}-{index}", label=label))
    seen = _row("t2-newmetric", label=2, metric="other", answer="a-t2-0")
    seen["question_key"] = "q-t2"
    unseen = _row("u-0", label=1, metric="m", raters=(1, 2, 3))
    rows = annotate_dev(train, [seen, unseen])
    assert rows[0]["response_seen"] is True
    assert rows[0]["exact_response_metric_edge_seen"] is False
    assert rows[1]["response_seen"] is False
    assert rows[1]["supported"] is True
    assert rows[1]["ambiguity"] == "H2"


def test_adjacent_boundary_support_is_conservative() -> None:
    train = [_row(f"a-{i}", label=1) for i in range(25)]
    train += [_row(f"b-{i}", label=2) for i in range(2)]
    train += [_row(f"c-{i}", label=3) for i in range(25)]
    train += [_row(f"d-{i}", label=4) for i in range(25)]
    train += [_row(f"e-{i}", label=5) for i in range(25)]
    annotated = annotate_dev(train, [_row("u-1", label=2)])[0]
    assert annotated["boundary"]["1"]["support"] == 25
    assert annotated["boundary"]["2"]["support"] == 27
    assert annotated["adjacent_support"] == 25
    assert annotated["supported"] is True


def test_score_metrics_report_numerators_denominators_and_probability_metrics() -> None:
    labels = [1, 2, 4, 5]
    predictions = [4, 2, 2, 5]
    probabilities = np.eye(5)[np.asarray(predictions) - 1] * 0.95 + 0.01
    metrics = score_metrics(labels, predictions, probabilities)
    assert metrics["L2H_numerator"] == 1
    assert metrics["L2H_denominator"] == 2
    assert metrics["H2L_numerator"] == 1
    assert metrics["H2L_denominator"] == 2
    assert metrics["Recall_2"] == 1.0
    assert metrics["NLL"] > 0
    assert metrics["RPS"] >= 0
    assert set(metrics["classwise_ECE"]) == {"1", "2", "3", "4", "5"}
    assert metrics["low_tail_ECE"] >= 0


def test_landing_report_detects_middle_capture() -> None:
    rows = [
        {
            "label": 2,
            "response_seen": True,
            "supported": True,
            "ambiguity": "H0",
            "metric_id": "m",
            "language": "en",
        },
        {
            "label": 2,
            "response_seen": False,
            "supported": True,
            "ambiguity": "H0",
            "metric_id": "m",
            "language": "en",
        },
    ]
    r3 = np.asarray([[0.05, 0.10, 0.30, 0.40, 0.15], [0.05, 0.10, 0.30, 0.40, 0.15]])
    p1 = np.asarray([[0.05, 0.15, 0.47, 0.23, 0.10], [0.05, 0.15, 0.47, 0.23, 0.10]])
    report = landing_report(rows, r3, p1)
    assert report["eligible_n"] == 1
    assert math.isclose(report["GLE"], 0.05 / 0.22)
    assert math.isclose(report["MCR"], 0.17 / 0.22)


def test_boundary_cell_association_uses_cell_means() -> None:
    rows = []
    predictions = []
    for metric, supports, errors in (
        ("m1", [1, 2, 3, 4], [1, 1, 0, 0]),
        ("m2", [2, 3, 4, 5], [1, 0, 0, 0]),
    ):
        for repeat in range(3):
            rows.append(
                {
                    "metric_id": metric,
                    "language": "en",
                    "response_seen": True,
                    "label": 3,
                    "boundary": {str(k): {"support": supports[k - 1]} for k in range(1, 5)},
                }
            )
            # One prediction creates errors on low-support lower boundaries.
            predictions.append(1 if repeat < 2 else 3)
    result = boundary_cell_support_association(rows, predictions, exposure=True)
    assert result["cell_n"] == 8
    assert result["raw_spearman_rho"] < 0


def test_cluster_bootstrap_preserves_question_dependence() -> None:
    rows = [
        {"question_key": "q1", "label": 1, "bad": True},
        {"question_key": "q1", "label": 1, "bad": True},
        {"question_key": "q2", "label": 1, "bad": False},
        {"question_key": "q2", "label": 1, "bad": False},
    ]
    report = clustered_gap_bootstrap(rows, [3, 3, 1, 1], lambda row: row["bad"], replicates=200, seed=7)
    assert report["estimate"] == 2.0
    assert report["ci95"][0] >= 0


def test_holm_adjustment_is_monotone() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5, "missing": None})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] >= adjusted["a"]
    assert adjusted["c"] >= adjusted["b"]
    assert adjusted["missing"] is None


def test_leave_one_group_out_gap_reports_stability() -> None:
    rows = [
        {"metric_id": "m1", "label": 1, "bad": True},
        {"metric_id": "m1", "label": 1, "bad": False},
        {"metric_id": "m2", "label": 1, "bad": True},
        {"metric_id": "m2", "label": 1, "bad": False},
    ]
    result = leave_one_group_out_gaps(rows, [3, 1, 2, 1], field="metric_id", adverse_group=lambda row: row["bad"])
    assert result["held_out_values"]["m1"] == 1.0
    assert result["held_out_values"]["m2"] == 2.0
    assert result["all_same_positive_direction"] is True


def test_prediction_map_rejects_reorder_and_parse_failure() -> None:
    rows = [
        {"record_id": "r1", "row_position": 0, "parse_success": True, "prediction": {"score": 2}},
        {"record_id": "r2", "row_position": 1, "parse_success": True, "prediction": {"score": 3}},
    ]
    assert _prediction_map(rows, ["r1", "r2"]) == {"r1": 2, "r2": 3}
    with pytest.raises(ValueError):
        _prediction_map(list(reversed(rows)), ["r1", "r2"])
    rows[0]["parse_success"] = False
    with pytest.raises(ValueError):
        _prediction_map(rows, ["r1", "r2"])


def test_ambiguity_categories_are_exhaustive() -> None:
    assert _ambiguity(_row("r0", label=2, raters=(2, 2, 2))) == "H0"
    assert _ambiguity(_row("r1", label=2, raters=(2, 2, 3))) == "H1"
    assert _ambiguity(_row("r2", label=2, raters=(1, 2, 3))) == "H2"


def test_public_formal_report_and_lock_audit() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    assert audit(
        root / "thesis_exp/outputs/exp54_rar_sft/rar_v2/residual_risk_decomposition_v1/public_report.json",
        root / "thesis_exp/outputs/exp54_rar_sft/rar_v2/residual_risk_decomposition_v1/public_lock.json",
        root / "thesis_exp/exp54_rar_sft/analyze_residual_risk_decomposition.py",
        root / "thesis_exp/exp54_rar_sft/configs/residual_risk_decomposition_v1.json",
        root / "thesis_exp/tests/test_exp54_residual_risk_decomposition.py",
    ) == "RESIDUAL_RISK_DECOMPOSITION_PASS"
