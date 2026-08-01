from __future__ import annotations

import math

import numpy as np

from thesis_exp.exp54_rar_sft.audit_common_support_exposure_nested_landing import audit
from thesis_exp.exp54_rar_sft.analyze_common_support_exposure_nested_landing import (
    add_direct_adjacent_support,
    common_support_standardized_contrast,
    exposure_level,
    nested_probability_transport,
)


def _annotated(level: str, *, label: int, question: str, metric: str = "m", ambiguity: str = "H0") -> dict:
    return {
        "question_key": question,
        "metric_id": metric,
        "language": "en",
        "label": label,
        "ambiguity": ambiguity,
        "response_seen": level == "E2",
        "question_seen": level in {"E1", "E2"},
        "supported": False,
        "direct_adjacent_supported": False,
    }


def test_exposure_levels_are_mutually_exclusive() -> None:
    assert exposure_level(_annotated("E2", label=4, question="q")) == "E2"
    assert exposure_level(_annotated("E1", label=4, question="q")) == "E1"
    assert exposure_level(_annotated("E0", label=4, question="q")) == "E0"


def test_common_support_standardization_removes_label_composition() -> None:
    rows = [
        _annotated("E1", label=4, question="q1"),
        _annotated("E2", label=4, question="q2"),
        _annotated("E1", label=5, question="q3"),
        _annotated("E2", label=5, question="q4"),
        _annotated("E2", label=1, question="q5"),
    ]
    # The unmatched label-1 E2 row has a huge error but must not enter the
    # common-support contrast.
    result = common_support_standardized_contrast(
        rows,
        [1.0, 0.0, 0.0, 1.0, 100.0],
        left="E1",
        right="E2",
        replicates=200,
        seed=3,
    )
    assert result["labels_in_common_support"] == [4, 5]
    assert result["left_n"] == 2 and result["right_n"] == 2
    assert math.isclose(result["estimate"], 0.0)


def test_common_support_reports_unidentified_empty_level() -> None:
    rows = [_annotated("E2", label=4, question="q1")]
    result = common_support_standardized_contrast(rows, [0.0], left="E0", right="E2", replicates=10, seed=1)
    assert result["identified"] is False
    assert result["estimate"] is None


def test_direct_adjacent_support_counts_exact_classes() -> None:
    train = []
    for label, count in ((1, 25), (2, 2), (3, 25), (4, 25), (5, 25)):
        train.extend({"metric_id": "m", "language": "en", "label_5": label} for _ in range(count))
    rows = [_annotated("E2", label=2, question="q")]
    add_direct_adjacent_support(train, rows)
    assert rows[0]["direct_adjacent_support"] == {"1": 2, "2": 2}
    assert rows[0]["direct_adjacent_supported"] is False


def test_nested_transport_reports_complete_mass_partition() -> None:
    rows = [
        _annotated("E2", label=2, question="q1"),
        _annotated("E1", label=2, question="q2"),
    ]
    rows[0]["supported"] = True
    rows[0]["direct_adjacent_supported"] = True
    r3 = np.asarray([[0.05, 0.10, 0.30, 0.40, 0.15], [0.05, 0.10, 0.30, 0.40, 0.15]])
    p1 = np.asarray([[0.10, 0.15, 0.47, 0.18, 0.10], [0.10, 0.15, 0.47, 0.18, 0.10]])
    report = nested_probability_transport(rows, r3, p1)
    assert report["all_low"]["positive_delta_high_n"] == 2
    assert report["response_seen"]["positive_delta_high_n"] == 1
    assert report["cumulative_supported"]["positive_delta_high_n"] == 1
    assert math.isclose(report["all_low"]["capture_partition_sum"], 1.0)
    assert abs(report["all_low"]["probability_conservation_residual"]) < 1e-12


def test_formal_patch_report_and_lock_pass() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    assert audit(
        root / "thesis_exp/outputs/exp54_rar_sft/rar_v2/common_support_exposure_nested_landing_patch_v1/public_report.json",
        root / "thesis_exp/outputs/exp54_rar_sft/rar_v2/common_support_exposure_nested_landing_patch_v1/public_lock.json",
        root / "thesis_exp/exp54_rar_sft/analyze_common_support_exposure_nested_landing.py",
        root / "thesis_exp/exp54_rar_sft/configs/common_support_exposure_nested_landing_patch_v1.json",
        root / "thesis_exp/tests/test_exp54_common_support_exposure_nested_landing.py",
    ) == "COMMON_SUPPORT_EXPOSURE_NESTED_LANDING_PASS"
