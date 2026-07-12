#!/usr/bin/env python3
"""Validate Exp33B DRGA artifacts and hard boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp33_direction_aware_aggregation/outputs/exp33b_direction_aware_aggregation_seed42"
)
DEFAULT_CONFIG = Path(
    "thesis_exp/exp33_direction_aware_aggregation/configs/exp33b_drga_preregistration_config.json"
)

REQUIRED_PUBLIC = (
    "configs/exp33b_drga_preregistration_config.json",
    "tables/exp33b_crossfit_metrics.csv",
    "tables/exp33b_source_reliability.csv",
    "tables/exp33b_fold_balance.csv",
    "tables/exp33b_direction_flags.csv",
    "tables/exp33b_risk_stress_metrics.csv",
    "tables/exp33b_train_supervision_public_summary.csv",
    "reports/exp33b_direction_aware_aggregation_report.md",
    "decision/exp33b_direction_aware_aggregation_decision.json",
    "hashes/exp33b_input_hashes.json",
    "hashes/exp33b_artifact_hashes.json",
)
METRIC_FIELDS = (
    "mae",
    "qwk",
    "exact",
    "within_one",
    "signed_bias",
    "severe_error",
    "low_to_high",
    "high_to_low",
    "label1_recall",
    "label2_recall",
    "label5_recall",
    "brier",
    "log_loss",
    "ece",
)


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def guarded(path: Path) -> Path:
    absolute = repo_path(path)
    if absolute.name.casefold() in {"dev.jsonl", "test.jsonl"}:
        raise PermissionError(f"Exp33B validator forbids reading nontrain split: {absolute}")
    return absolute


def read_json(path: Path) -> dict[str, Any]:
    with guarded(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with guarded(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.rows.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    def run(self, name: str, function: Callable[[], tuple[bool, str]]) -> None:
        try:
            passed, detail = function()
        except Exception as exc:  # collect failures together
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        self.add(name, bool(passed), detail)

    @property
    def passed(self) -> bool:
        return all(row["status"] == "PASS" for row in self.rows)


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def check_required_files(out_dir: Path) -> tuple[bool, str]:
    missing = [relative for relative in REQUIRED_PUBLIC if not guarded(out_dir / relative).is_file()]
    return not missing, f"missing={missing or 'none'}"


def check_config_lock(config_path: Path, out_dir: Path) -> tuple[bool, str]:
    source = read_json(config_path)
    copied = read_json(out_dir / "configs/exp33b_drga_preregistration_config.json")
    failures = []
    if source != copied:
        failures.append("output_config_differs_from_preregistered_config")
    if source.get("method_short_name") != "DRGA":
        failures.append("method_short_name")
    if int(source.get("folds", 0)) != 5:
        failures.append("folds")
    forbidden = set(source.get("forbidden_inputs") or [])
    if not {"dev.jsonl", "test.jsonl"}.issubset(forbidden):
        failures.append("forbidden_inputs")
    return not failures, f"failures={failures or 'none'}"


def check_decision(out_dir: Path) -> tuple[bool, str]:
    decision = read_json(out_dir / "decision/exp33b_direction_aware_aggregation_decision.json")
    failures = []
    expectations = {
        "reference_status": "model-reviewed silver, not human expert gold",
        "risk_view_used_for_prevalence": False,
        "dev_rows_read": 0,
        "test_access_count": 0,
        "clean_dev_read": False,
        "api_called": False,
        "gpu_used": False,
        "training_run": False,
        "student_inference_run": False,
        "downstream_student_metric_tuning": False,
    }
    for key, expected in expectations.items():
        if decision.get(key) != expected:
            failures.append(f"{key}={decision.get(key)!r}")
    if int(decision.get("calibration_rows", 0)) != 120:
        failures.append("calibration_rows")
    if int(decision.get("crossfit_folds", 0)) != 5:
        failures.append("crossfit_folds")
    gates = decision.get("quality_gates")
    if not isinstance(gates, dict) or sorted(gates) != sorted(
        [
            "drga_not_worse_than_rounded_human_mae",
            "drga_not_worse_than_rounded_human_qwk",
            "drga_not_worse_low_to_high",
            "drga_not_worse_high_to_low",
            "posterior_valid",
            "fallback_recomputable",
            "no_dev_or_test_access",
        ]
    ):
        failures.append("quality_gates_shape")
    if decision.get("quality_gate_passed") and int(decision.get("full_train_supervision_rows", 0)) != 2654:
        failures.append("supervision_rows_when_gate_passed")
    if not decision.get("quality_gate_passed") and decision.get("full_train_supervision_generated"):
        failures.append("supervision_generated_when_gate_failed")
    return not failures, f"failures={failures or 'none'}"


def check_metrics(out_dir: Path) -> tuple[bool, str]:
    rows = read_csv(out_dir / "tables/exp33b_crossfit_metrics.csv")
    by_method = {row.get("method"): row for row in rows}
    required = {
        "rounded_human",
        "human_median",
        "human_mean",
        "qwen",
        "deepseek",
        "teacher_mean",
        "teacher_median",
        "Dawid-Skene",
        "MACE",
        "equal_weight_fusion",
        "DRGA",
    }
    failures = []
    missing = sorted(required - set(by_method))
    if missing:
        failures.append(f"missing_methods={missing}")
    for method in required & set(by_method):
        row = by_method[method]
        if int(float(row.get("rows") or 0)) <= 0:
            failures.append(f"{method}_rows")
        if row.get("weighting") != "design_weighted":
            failures.append(f"{method}_weighting")
        for field in METRIC_FIELDS:
            value = parse_float(row.get(field))
            if value is None or not math.isfinite(value):
                if field.startswith("label"):
                    continue
                failures.append(f"{method}_{field}")
                break
    if "DRGA" in by_method and "rounded_human" in by_method:
        drga = by_method["DRGA"]
        rounded = by_method["rounded_human"]
        decision = read_json(out_dir / "decision/exp33b_direction_aware_aggregation_decision.json")
        gates = decision["quality_gates"]
        comparisons = {
            "drga_not_worse_than_rounded_human_mae": parse_float(drga["mae"]) <= parse_float(rounded["mae"]) + 1e-12,
            "drga_not_worse_than_rounded_human_qwk": parse_float(drga["qwk"]) >= parse_float(rounded["qwk"]) - 1e-12,
            "drga_not_worse_low_to_high": parse_float(drga["low_to_high"]) <= parse_float(rounded["low_to_high"]) + 1e-12,
            "drga_not_worse_high_to_low": parse_float(drga["high_to_low"]) <= parse_float(rounded["high_to_low"]) + 1e-12,
        }
        for key, actual in comparisons.items():
            if bool(gates.get(key)) != bool(actual):
                failures.append(f"gate_mismatch_{key}")
    return not failures, f"failures={failures or 'none'}"


def check_fold_balance(out_dir: Path) -> tuple[bool, str]:
    rows = read_csv(out_dir / "tables/exp33b_fold_balance.csv")
    fold_counts: dict[str, int] = {}
    total = 0
    for row in rows:
        fold = str(row["fold"])
        count = int(float(row["rows"]))
        fold_counts[fold] = fold_counts.get(fold, 0) + count
        total += count
    failures = []
    if total != 120:
        failures.append(f"total={total}")
    if sorted(fold_counts) != ["0", "1", "2", "3", "4"]:
        failures.append(f"folds={sorted(fold_counts)}")
    if max(fold_counts.values() or [0]) - min(fold_counts.values() or [0]) > 8:
        failures.append(f"imbalance={fold_counts}")
    return not failures, f"fold_counts={fold_counts}; failures={failures or 'none'}"


def check_private_boundary(out_dir: Path) -> tuple[bool, str]:
    decision = read_json(out_dir / "decision/exp33b_direction_aware_aggregation_decision.json")
    public_text = ""
    for relative in REQUIRED_PUBLIC:
        path = guarded(out_dir / relative)
        public_text += path.read_text(encoding="utf-8")
    failures = []
    forbidden_public_terms = [
        "adjudication_reason",
        "review_reason",
        "qwen_reason",
        "deepseek_reason",
        "evaluator_output",
        "question_context",
        "raw reason",
    ]
    for term in forbidden_public_terms:
        if term in public_text:
            failures.append(f"public_contains_{term}")
    private_supervision = out_dir / "private/exp33b_train_supervision.csv"
    if decision.get("quality_gate_passed"):
        if not guarded(private_supervision).is_file():
            failures.append("missing_private_supervision")
        else:
            result = subprocess.run(
                ["git", "check-ignore", "--quiet", str(guarded(private_supervision))],
                cwd=REPO_ROOT,
                check=False,
            )
            if result.returncode != 0:
                failures.append("private_supervision_not_gitignored")
    private_predictions = out_dir / "private/exp33b_representative_crossfit_predictions.csv"
    if not guarded(private_predictions).is_file():
        failures.append("missing_private_crossfit_predictions")
    else:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", str(guarded(private_predictions))],
            cwd=REPO_ROOT,
            check=False,
        )
        if result.returncode != 0:
            failures.append("private_predictions_not_gitignored")
    return not failures, f"failures={failures or 'none'}"


def check_private_predictions(out_dir: Path) -> tuple[bool, str]:
    rows = read_csv(out_dir / "private/exp33b_representative_crossfit_predictions.csv")
    metric_rows = read_csv(out_dir / "tables/exp33b_crossfit_metrics.csv")
    expected_rows = sum(int(float(row["rows"])) for row in metric_rows if row.get("status") == "COMPLETE")
    failures = []
    if len(rows) != expected_rows:
        failures.append(f"rows={len(rows)} expected={expected_rows}")
    for index, row in enumerate(rows[:]):
        probs = [parse_float(row.get(f"p{label}")) for label in range(1, 6)]
        if any(value is None for value in probs) or abs(sum(float(value) for value in probs) - 1.0) > 1e-8:
            failures.append(f"posterior_{index}")
            break
        label = int(float(row["hard_label"]))
        if not 1 <= label <= 5:
            failures.append(f"hard_label_{index}")
            break
    return not failures, f"rows={len(rows)}; failures={failures or 'none'}"


def check_supervision(out_dir: Path) -> tuple[bool, str]:
    decision = read_json(out_dir / "decision/exp33b_direction_aware_aggregation_decision.json")
    path = out_dir / "private/exp33b_train_supervision.csv"
    if not decision.get("quality_gate_passed"):
        return not guarded(path).exists(), "gate failed; private supervision absent as expected"
    rows = read_csv(path)
    failures = []
    if len(rows) != 2654:
        failures.append(f"rows={len(rows)}")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        failures.append("duplicate_sample_id")
    for index, row in enumerate(rows):
        probs = [parse_float(row.get(f"p{label}")) for label in range(1, 6)]
        if any(value is None for value in probs) or abs(sum(float(value) for value in probs) - 1.0) > 1e-8:
            failures.append(f"posterior_{index}")
            break
        weight = parse_float(row.get("sample_weight"))
        if weight is None or not (0.0 < weight <= 1.0):
            failures.append(f"sample_weight_{index}")
            break
        if int(float(row["hard_label"])) not in {1, 2, 3, 4, 5}:
            failures.append(f"hard_label_{index}")
            break
    return not failures, f"rows={len(rows)}; failures={failures or 'none'}"


def check_risk_status(out_dir: Path) -> tuple[bool, str]:
    decision = read_json(out_dir / "decision/exp33b_direction_aware_aggregation_decision.json")
    rows = read_csv(out_dir / "tables/exp33b_risk_stress_metrics.csv")
    status = decision.get("risk_silver_status", {}).get("status")
    failures = []
    if status != "COMPLETE":
        if not all(row.get("status") == "PENDING_RISK_SILVER" and int(float(row.get("rows") or 0)) == 0 for row in rows):
            failures.append("risk_pending_rows_not_explicit")
    else:
        if not all(row.get("weighting") == "unweighted" for row in rows):
            failures.append("risk_weighting")
    return not failures, f"risk_status={status}; failures={failures or 'none'}"


def check_heavy(out_dir: Path) -> tuple[bool, str]:
    decision = read_json(out_dir / "decision/exp33b_direction_aware_aggregation_decision.json")
    hashes = read_json(out_dir / "hashes/exp33b_input_hashes.json")
    failures = []
    train = hashes.get("train_split", {})
    if train.get("path", "").endswith("train.jsonl") is False:
        failures.append("train_hash_path")
    for key in ("exp33a_reviewer_a", "exp33a_reviewer_b", "exp33a_adjudicator"):
        if key not in hashes or len(str(hashes[key].get("sha256", ""))) != 64:
            failures.append(f"{key}_hash")
    for item in hashes.get("teachers", []):
        if len(str(item.get("sha256", ""))) != 64:
            failures.append("teacher_hash")
    if "dev" in json.dumps(hashes).lower() or "test.jsonl" in json.dumps(hashes).lower():
        failures.append("hashes_reference_forbidden_split")
    if decision.get("full_train_supervision_generated"):
        private_hash = decision.get("private_supervision_sha256")
        if not private_hash or len(str(private_hash)) != 64:
            failures.append("private_supervision_hash")
    return not failures, f"failures={failures or 'none'}"


def validate(args: argparse.Namespace) -> Checks:
    out_dir = args.out_dir
    checks = Checks()
    checks.run("required_public_files", lambda: check_required_files(out_dir))
    checks.run("preregistered_config_lock", lambda: check_config_lock(args.config, out_dir))
    checks.run("decision_boundaries", lambda: check_decision(out_dir))
    checks.run("crossfit_metrics_and_gates", lambda: check_metrics(out_dir))
    checks.run("fold_balance", lambda: check_fold_balance(out_dir))
    checks.run("private_boundary", lambda: check_private_boundary(out_dir))
    checks.run("private_predictions", lambda: check_private_predictions(out_dir))
    checks.run("supervision_gate_and_rows", lambda: check_supervision(out_dir))
    checks.run("risk_status", lambda: check_risk_status(out_dir))
    if args.heavy:
        checks.run("heavy_hash_and_input_boundary", lambda: check_heavy(out_dir))
    write_csv(
        out_dir / "tables/exp33b_validation_checks.csv",
        checks.rows,
        ["check", "status", "detail"],
    )
    report = ["# Exp33B Validation Report", ""]
    report.extend(f"- {row['check']}: {row['status']} ({row['detail']})" for row in checks.rows)
    (repo_path(out_dir) / "reports/exp33b_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--heavy", action="store_true")
    return parser.parse_args()


def main() -> None:
    checks = validate(parse_args())
    print(json.dumps({"status": "PASS" if checks.passed else "FAIL", "checks": len(checks.rows)}, sort_keys=True))
    if not checks.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
