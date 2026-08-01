from __future__ import annotations

from pathlib import Path

from thesis_exp.exp54_rar_sft.analyze_crossed_metric_conditional_fidelity_feasibility import (
    crossed_inventory,
    feasibility_report,
)
from thesis_exp.exp54_rar_sft.audit_crossed_metric_conditional_fidelity_feasibility import audit


def _row(qa: str, metric: str) -> dict[str, str]:
    return {"question_key": qa, "answer_key": f"a-{qa}", "metric_id": metric}


def _protocol(*, dev_nodes: int = 1, train_pair: int = 1, dev_pair: int = 1, pairs: int = 1) -> dict:
    return {
        "go_requirements": {
            "minimum_multi_metric_dev_qa_nodes": dev_nodes,
            "minimum_train_shared_qa_per_metric_pair": train_pair,
            "minimum_dev_shared_qa_per_metric_pair": dev_pair,
            "minimum_jointly_eligible_metric_pairs": pairs,
        }
    }


def test_duplicate_edge_does_not_inflate_pair_support() -> None:
    rows = [_row("q1", "m1"), _row("q1", "m1"), _row("q1", "m2")]
    inventory = crossed_inventory(rows)
    assert inventory["pair_counts"][("m1", "m2")] == 1
    assert inventory["duplicate_response_metric_edge_count"] == 1
    assert inventory["duplicate_extra_row_count"] == 1


def test_joint_pair_gate_requires_support_in_both_splits() -> None:
    train = [_row(f"q{i}", metric) for i in range(3) for metric in ("m1", "m2")]
    dev = [_row("d0", "m1"), _row("d0", "m2")]
    report = feasibility_report(train, dev, _protocol(train_pair=3, dev_pair=2))
    assert report["gate_results"]["multi_metric_dev_qa_nodes"]["passed"] is True
    assert report["gate_results"]["jointly_eligible_metric_pairs"]["observed"] == 0
    assert report["decision"] == "NO_GO_INSUFFICIENT_CROSSED_METRIC_SUPPORT"


def test_feasible_synthetic_inventory_returns_go() -> None:
    rows = [_row(f"q{i}", metric) for i in range(3) for metric in ("m1", "m2")]
    report = feasibility_report(rows, rows, _protocol(train_pair=3, dev_pair=3))
    assert report["gate_results"]["jointly_eligible_metric_pairs"]["observed"] == 1
    assert report["decision"] == "GO_FULL_CONDITIONAL_FIDELITY_AUDIT"


def test_formal_feasibility_report_and_lock_pass() -> None:
    root = Path(__file__).resolve().parents[2]
    base = root / "thesis_exp/outputs/exp54_rar_sft/rar_v2/crossed_metric_conditional_fidelity_feasibility_v1"
    assert audit(
        root / "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl",
        root / "thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl",
        root / "thesis_exp/exp54_rar_sft/configs/crossed_metric_conditional_fidelity_feasibility_v1.json",
        base / "public_report.json",
        base / "public_lock.json",
        root / "thesis_exp/exp54_rar_sft/analyze_crossed_metric_conditional_fidelity_feasibility.py",
        root / "thesis_exp/tests/test_exp54_crossed_metric_conditional_fidelity_feasibility.py",
    ) == "CROSSED_METRIC_CONDITIONAL_FIDELITY_FEASIBILITY_PASS"
