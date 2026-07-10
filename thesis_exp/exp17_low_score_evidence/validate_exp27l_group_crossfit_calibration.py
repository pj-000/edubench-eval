"""Validate Exp27L protocol boundaries and lightweight output integrity."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.exp27l_common import DEFAULT_OUT, read_csv, read_jsonl  # noqa: E402


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def validate(args: argparse.Namespace) -> dict[str, object]:
    out = args.out_dir
    required = [
        out / "data" / "exp27l_question_key_fold_assignment.csv",
        out / "tables" / "exp27l_fold_balance.csv",
        out / "tables" / "exp27l_leakage_audit.csv",
        out / "decision" / "exp27l_prepare_decision.json",
        out / "data" / "exp27l_oof_soft_score_predictions.csv",
        out / "data" / "exp27l_oof_risk_predictions.csv",
        out / "tables" / "exp27l_selected_calibration_parameters.csv",
        out / "tables" / "exp27l_cluster_bootstrap_ci.csv",
        out / "decision" / "exp27l_group_crossfit_decision.json",
        out / "annotation_templates" / "exp27l_external_blind_review_template.csv",
        out / "decision" / "exp27l_external_lockbox_decision.json",
    ]
    for path in required:
        require(path)

    assignments = read_csv(out / "data" / "exp27l_question_key_fold_assignment.csv")
    if len(assignments) != 180 or len({row["sample_id"] for row in assignments}) != 180:
        raise ValueError("Expected exactly 180 unique fold-assignment sample IDs")
    folds_by_qhash: dict[str, set[str]] = defaultdict(set)
    for row in assignments:
        folds_by_qhash[row["question_key_hash"]].add(row["outer_fold"])
    split_qkeys = [key for key, folds in folds_by_qhash.items() if len(folds) != 1]
    if split_qkeys:
        raise ValueError(f"Question-key leakage across outer folds: {split_qkeys[:5]}")
    if set(row["outer_fold"] for row in assignments) != {"0", "1", "2", "3", "4"}:
        raise ValueError("Expected all five outer folds")
    counts = Counter(row["view"] for row in assignments)
    if counts != {"representative": 120, "risk_enriched": 60}:
        raise ValueError(f"Unexpected view counts: {counts}")

    leakage = {row["check"]: int(row["count"]) for row in read_csv(out / "tables" / "exp27l_leakage_audit.csv")}
    for key in (
        "inherited_exp27j_dev_sample_overlap",
        "inherited_exp27j_dev_question_overlap",
        "inherited_exp27j_test_sample_overlap",
        "inherited_exp27j_test_question_overlap",
        "dev_test_files_opened_by_exp27l_prepare",
        "teacher_api_calls",
        "gpu_training_runs",
    ):
        if leakage.get(key) != 0:
            raise ValueError(f"Leakage/protocol check failed: {key}={leakage.get(key)}")

    score_rows = read_csv(out / "data" / "exp27l_oof_soft_score_predictions.csv")
    risk_rows = read_csv(out / "data" / "exp27l_oof_risk_predictions.csv")
    if len(score_rows) != 180 or len(risk_rows) != 180:
        raise ValueError("Expected one OOF score and risk row for all 180 samples")
    if {row["sample_id"] for row in score_rows} != {row["sample_id"] for row in risk_rows}:
        raise ValueError("Score and risk OOF IDs do not align")
    parameters = read_csv(out / "tables" / "exp27l_selected_calibration_parameters.csv")
    if len(parameters) != 5 or any(row["risk_fit_population"] != "representative_only" for row in parameters):
        raise ValueError("Every outer risk fit must use representative rows only")
    if any(row["risk_stress_fit_rows"] != "0" for row in parameters):
        raise ValueError("Risk-stress rows entered a risk fit")
    bootstrap = read_csv(out / "tables" / "exp27l_cluster_bootstrap_ci.csv")
    if not bootstrap or any(row["resamples"] != "2000" for row in bootstrap):
        raise ValueError("Exp27L needs 2000 question-key cluster bootstrap resamples")

    for path in (out / "decision" / "exp27l_prepare_decision.json", out / "decision" / "exp27l_group_crossfit_decision.json"):
        decision = json.loads(path.read_text(encoding="utf-8"))
        if decision.get("proceed_to_exp27m_train") is not False or decision.get("proceed_to_reranker_training") is not False:
            raise ValueError(f"Training gate must remain false in {path.name}")
    lockbox = json.loads((out / "decision" / "exp27l_external_lockbox_decision.json").read_text(encoding="utf-8"))
    if lockbox.get("external_review_complete") is not False or lockbox.get("proceed_to_exp27m_train") is not False:
        raise ValueError("External lockbox must remain incomplete and non-training")

    template_rows = read_csv(out / "annotation_templates" / "exp27l_external_blind_review_template.csv")
    prohibited_columns = {"original_score", "qwen_score", "deepseek_score", "silver_final_score", "risk_probability", "exp27i_v1_tier"}
    found = prohibited_columns & set(template_rows[0]) if template_rows else set()
    if found:
        raise ValueError(f"Blind review template exposes prohibited fields: {sorted(found)}")
    packets = out / "packets" / "exp27l_external_blind_core_packets.jsonl"
    if packets.exists():
        for packet in read_jsonl(packets):
            found = prohibited_columns & set(packet)
            if found:
                raise ValueError(f"Blind packet exposes prohibited top-level fields: {sorted(found)}")

    return {
        "status": "PASS",
        "fold_assignments": len(assignments),
        "oof_rows": len(score_rows),
        "outer_question_key_leakage": 0,
        "risk_stress_fit_rows": 0,
        "bootstrap_resamples": 2000,
        "training_gate": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), ensure_ascii=False, sort_keys=True))
